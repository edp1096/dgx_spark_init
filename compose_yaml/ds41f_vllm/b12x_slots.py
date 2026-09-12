"""Fixed GPU expert slots, parallel packed-file reads and replayable b12x MoE."""
from collections import OrderedDict, Counter, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,os,struct,time,math,threading
from contextlib import nullcontext
from pathlib import Path
import torch
from b12x.moe import fused_moe
from b12x_layout import FORMAT,blank,tensors
import step_profile
import expert_io
import cache_layout
import probe_trace

HEADER_BYTES=65536
_pool=ThreadPoolExecutor(max_workers=int(os.environ.get('DSV41_READ_THREADS','4')))
_layers={}
_arenas={}
_io_buffers={}

def execution_options():
    if os.environ.get("DSV41_BENCH_CONTROL")=="1":
        import cache_control
        return cache_control.kernel_tokens,cache_control.shared_buffers
    return int(os.environ.get("DSV41_KERNEL_TOKENS","512")),os.environ.get("DSV41_SHARED_BUFFERS","0")=="1"

def execution_buffers(m,topk,shared):
    # Graph outputs are borrowed until the next call with this key. Consumers
    # must enqueue their copy/add on this stream before another layer replays.
    key=(torch.cuda.current_device(),torch.cuda.current_stream().cuda_stream,m,topk)
    if shared and key in _io_buffers: return _io_buffers[key]
    buffers={"x":torch.zeros(m,5120,device="cuda",dtype=torch.bfloat16),
             "ids":torch.zeros(m,topk,device="cuda",dtype=torch.int32),
             "weights":torch.zeros(m,topk,device="cuda",dtype=torch.float32),
             "out":torch.empty(m,5120,device="cuda",dtype=torch.float32)}
    if shared: _io_buffers[key]=buffers
    return buffers
_readers=threading.local()
_prefetch_pool=ThreadPoolExecutor(max_workers=2)
_prefetch_tickets=set()

def shared_scratch(plan):
    # Model layers execute sequentially on a stream. Captured graphs retain
    # views into one stable arena instead of reserving ~80-130 MiB PER layer
    # and PER token shape. The arena is never resized after graph capture.
    key=(torch.cuda.current_device(),torch.cuda.current_stream().cuda_stream)
    if key not in _arenas:
        size=int(os.environ.get('DSV41_SCRATCH_MIB','256'))*1024**2
        _arenas[key]=torch.empty(size,dtype=torch.uint8,device='cuda')
    arena=_arenas[key];offset=0;result={}
    for spec in plan.scratch_specs():
        offset=(offset+255)//256*256
        size=math.prod(spec.shape)*spec.dtype.itemsize
        if offset+size>arena.numel():
            raise RuntimeError('Increase DSV41_SCRATCH_MIB before boot; live graph arenas cannot resize')
        result[spec.name]=arena[offset:offset+size].view(spec.dtype).reshape(spec.shape)
        offset+=size
    return result

_stats={'slot_hits':0,'slot_misses':0,'packed_read_bytes':0,'graph_replays':0,
        'routed_pipeline_batches':0,'routed_pipeline_bytes':0,
        'prefetch_issued':0,'prefetch_read_bytes':0,'prefetch_consumed':0,
        'prefetch_ready':0,'prefetch_late':0,'prefetch_cancelled':0,
        'prefetch_unused':0,'demand_read_bytes':0,'io_batches':0,'io_overlaps':0}

class SlotLayer:
    def __init__(self,store,layer,topk):
        self.layer=layer;self.topk=topk
        self.rank=store.rank
        self.path=Path(os.environ['DSV41_PACKED_DIR'])/f'layer-{layer:02d}.bin'
        with self.path.open('rb') as f:
            n=struct.unpack('<Q',f.read(8))[0]
            if not 0<n<HEADER_BYTES-8: raise ValueError('Bad packed header')
            self.meta=json.loads(f.read(n))
        if (self.meta['format'],self.meta['rank'],self.meta['revision'],self.meta['layer']) != (FORMAT,store.rank,store.model.name,layer):
            raise ValueError(f'Packed expert ABI/rank/revision mismatch: {self.path}')
        if self.path.stat().st_size!=HEADER_BYTES+self.meta['record_bytes']*self.meta['experts']:
            raise ValueError(f'Incomplete packed expert file: {self.path}')
        self.count=min(cache_layout.capacity(layer,int(os.environ.get('DSV41_SLOTS_PER_LAYER','128'))),self.meta['experts'])
        if self.count<topk: raise ValueError('Cache must fit at least one token routing set')
        reserve=(int(os.environ.get('DSV41_PREFETCH_SLOTS','2'))
                 if os.environ.get('DSV41_PREFETCH_TEST')=='1' and layer<40 else 0)
        if not 0<=reserve<=8: raise ValueError('Prefetch reserve must be between 0 and 8')
        self.capacity=self.count+reserve
        io_mode=os.environ.get('DSV41_SLOT_IO','buffered')
        if io_mode not in ('buffered','direct'):
            raise ValueError('DSV41_SLOT_IO must be buffered or direct')
        self.direct=io_mode=='direct'
        self.fd=os.open(self.path,os.O_RDONLY | (os.O_DIRECT if self.direct else 0))
        os.posix_fadvise(self.fd,0,0,os.POSIX_FADV_RANDOM)
        if self.direct:
            from b12x.loader._pool import shared_pool
        probe_trace.note('expert_slots_begin',layer=layer,slots=self.capacity)
        with shared_pool(allocation='registered') if self.direct else nullcontext():
            self.prepared=blank(self.capacity)
        probe_trace.note('expert_slots_end',layer=layer,slots=self.capacity)
        self.parts=tensors(self.prepared)
        for part,desc in zip(self.parts,self.meta['layout']):
            if list(part.shape[1:])!=desc['shape'] or str(part.dtype)!=desc['dtype']:
                raise ValueError('Prepared layout changed: rebuild expert files')
        self.used=OrderedDict();self.free=list(range(self.count-1,-1,-1))
        self.prefetch_free=list(range(self.count,self.capacity))
        self.prefetch_pending={}
        self.prefetch_stream=None
        self.calls={}
        self.last_event=torch.cuda.Event()
        self.last_stream=None

    def account_prefetch(self,item):
        item['future'].result()
        if not item['accounted']:
            _stats['prefetch_read_bytes']+=self.meta['record_bytes']
            _stats['packed_read_bytes']+=self.meta['record_bytes']
            item['accounted']=True

    def retire_prefetch(self,keep=(),wait=False):
        for expert,item in list(self.prefetch_pending.items()):
            if expert in keep: continue
            future=item['future']
            if future.cancel():
                _stats['prefetch_cancelled']+=1
            elif wait or future.done():
                self.account_prefetch(item)
                _stats['prefetch_unused']+=1
            else:
                continue
            self.prefetch_free.append(item['slot'])
            del self.prefetch_pending[expert]

    def reset(self):
        self.retire_prefetch(wait=True)
        self.used.clear()
        self.free=list(range(self.count-1,-1,-1))
        self.prefetch_free=list(range(self.count,self.capacity))

    def start_prefetch(self,experts):
        global _prefetch_tickets
        if not self.direct: raise ValueError('Expert prefetch requires direct shared storage')
        self.retire_prefetch(keep=experts)
        if self.prefetch_stream is None:
            self.prefetch_stream=torch.cuda.Stream()
        self.prefetch_stream.wait_stream(torch.cuda.current_stream())
        if self.last_stream is not None:
            self.prefetch_stream.wait_event(self.last_event)
        _prefetch_tickets={f for f in _prefetch_tickets if not f.done()}
        for expert in experts:
            if not 0<=expert<self.meta['experts']: raise ValueError('Invalid forecast expert ID')
            if expert in self.used or expert in self.prefetch_pending: continue
            if not self.prefetch_free or len(_prefetch_tickets)>=4: break
            slot=self.prefetch_free.pop()
            future=_prefetch_pool.submit(self.read_into,expert,slot,self.prefetch_stream.cuda_stream)
            self.prefetch_pending[expert]={'slot':slot,'future':future,'accounted':False}
            _prefetch_tickets.add(future)
            _stats['prefetch_issued']+=1

    def read(self,expert):
        length=self.meta['record_bytes']
        allocation=torch.empty(length+4096,dtype=torch.uint8,device='cpu',pin_memory=True)
        start=(-allocation.data_ptr())%4096
        data=allocation[start:start+length]
        view=memoryview(data.numpy())
        offset=HEADER_BYTES+expert*length;done=0
        while done<length:
            n=os.preadv(self.fd,[view[done:]],offset+done)
            if not n: raise EOFError(f'Short expert read: {self.path} expert {expert}')
            done+=n
        # The GPU slot is the explicit cache; avoid retaining another full
        # copy in Linux's page cache on the same unified-memory pool.
        os.posix_fadvise(self.fd,offset,length,os.POSIX_FADV_DONTNEED)
        if os.environ.get('DSV41_VERIFY_PACKED')=='1':
            if hashlib.blake2b(view,digest_size=16).hexdigest()!=self.meta['hashes'][expert]:
                raise ValueError('Packed expert checksum mismatch')
        return data

    def read_into(self,expert,slot,stream):
        # b12x verifies shared-pool ownership and waits for the prior consumer
        # before issuing O_DIRECT into the final, CPU-addressable CUDA storage.
        from b12x.loader._native import load
        native=load()
        if not hasattr(_readers,'reader'):
            _readers.reader=native.direct_reader(torch.cuda.current_device())
        offset=HEADER_BYTES+expert*self.meta['record_bytes']
        for part,desc in zip(self.parts,self.meta['layout']):
            native.direct_into(_readers.reader,self.fd,offset,desc['bytes'],
                               part[slot].data_ptr(),stream)
            offset+=desc['bytes']
        if os.environ.get('DSV41_VERIFY_PACKED')=='1':
            digest=hashlib.blake2b(digest_size=16)
            for part in self.parts:
                digest.update(memoryview(part[slot].cpu().contiguous().view(torch.uint8).numpy()))
            if digest.hexdigest()!=self.meta['hashes'][expert]:
                raise ValueError('Packed expert checksum mismatch')

    def ensure(self,needed,on_load_start=None):
        requested=set(needed)
        if not requested or min(requested)<0 or max(requested)>=self.meta['experts']:
            raise ValueError('Invalid routed expert ID')
        if len(requested)>self.count: raise ValueError('Routing set exceeds cache slots')
        self.retire_prefetch(keep=requested)
        missing=[]
        for expert in dict.fromkeys(needed):
            if expert in self.used:
                self.used.move_to_end(expert);_stats['slot_hits']+=1
            else:
                missing.append(expert)
        # Protect ALL experts used in this microbatch, including later hits.
        targets=[]
        for expert in missing:
            if self.free:
                slot=self.free.pop()
            else:
                victim=next(e for e in self.used if e not in requested)
                slot=self.used.pop(victim)
            prefetched=self.prefetch_pending.pop(expert,None)
            if prefetched is not None:
                self.prefetch_free.append(slot)
                slot=prefetched['slot']
                prefetched['ready']=prefetched['future'].done()
            targets.append((expert,slot,prefetched))
        io_mode=expert_io.mode()
        if io_mode!='serial' and (not self.direct or self.prefetch_pending):
            raise RuntimeError('Scheduled expert I/O requires direct storage without forecast reads')
        stream=torch.cuda.current_stream().cuda_stream
        if targets and on_load_start is not None:
            stream=expert_io.read_stream()
        if targets and io_mode in ('batch','batch_overlap'):
            if any(prefetched is not None for _,_,prefetched in targets):
                raise RuntimeError('Batch I/O cannot consume forecast reads')
            records=bytearray()
            for expert,slot,_ in targets:
                offset=HEADER_BYTES+expert*self.meta['record_bytes']
                for part,desc in zip(self.parts,self.meta['layout']):
                    records.extend(struct.pack('<8Q',self.fd,offset,desc['bytes'],
                        part[slot].data_ptr(),0,1,0,0))
                    offset+=desc['bytes']
            future=expert_io.submit_batch(records,stream)
            try:
                if on_load_start is not None:
                    _stats['io_overlaps']+=bool(on_load_start())
                future.result()
            except BaseException:
                # A callback failure must not leave writers using reused slots.
                try: future.result()
                except Exception: pass
                raise
            _stats['io_batches']+=1
            for expert,slot,_ in targets:
                self.used[expert]=slot
            _stats['slot_misses']+=len(targets)
            size=len(targets)*self.meta['record_bytes']
            _stats['packed_read_bytes']+=size
            _stats['demand_read_bytes']+=size
            if os.environ.get('DSV41_VERIFY_PACKED')=='1':
                for expert,slot,_ in targets: self.verify_slot(expert,slot)
            return [self.used[e] for e in needed]
        # Bound pinned staging. Keeping one Future for every requested expert
        # retains every completed 9.4 MB buffer until the whole wave ends.
        todo=iter(targets);pending=deque()
        def schedule():
            try: expert,slot,prefetched=next(todo)
            except StopIteration: return
            future=(prefetched['future'] if prefetched is not None else
                    _pool.submit(self.read_into,expert,slot,stream) if self.direct else
                    _pool.submit(self.read,expert))
            pending.append((expert,slot,future,prefetched))
        for _ in range(max(1,int(os.environ.get('DSV41_READ_THREADS','4'))*2)):
            schedule()
        if pending and on_load_start is not None:
            try:
                _stats['io_overlaps']+=bool(on_load_start())
            except BaseException:
                for _,_,future,_ in pending:
                    try: future.result()
                    except Exception: pass
                raise
        while pending:
            expert,slot,future,prefetched=pending.popleft()
            data=future.result();offset=0
            if not self.direct:
                for part,desc in zip(self.parts,self.meta['layout']):
                    length=desc['bytes']
                    source=data[offset:offset+length].view(part.dtype).reshape(desc['shape'])
                    # The synchronous copy also owns pinned staging lifetime.
                    part[slot].copy_(source,non_blocking=False)
                    offset+=length
                del source
            self.used[expert]=slot
            if prefetched is not None:
                self.account_prefetch(prefetched)
                _stats['prefetch_consumed']+=1
                _stats['prefetch_ready' if prefetched['ready'] else 'prefetch_late']+=1
                _stats['slot_hits']+=1
            else:
                _stats['slot_misses']+=1;_stats['packed_read_bytes']+=self.meta['record_bytes']
                _stats['demand_read_bytes']+=self.meta['record_bytes']
            del data,future
            schedule()
        return [self.used[e] for e in needed]

    def verify_slot(self,expert,slot):
        digest=hashlib.blake2b(digest_size=16)
        for part in self.parts:
            digest.update(memoryview(part[slot].cpu().contiguous().view(torch.uint8).numpy()))
        if digest.hexdigest()!=self.meta['hashes'][expert]:
            raise ValueError('Packed expert checksum mismatch')

    def execution(self,m):
        limit,shared=execution_options()
        m=next((n for n in (1,5,6,16,32,64,128,256,512,1024,2048) if n>=m and n<=limit),None)
        if m is None: raise ValueError("Expert execution exceeds configured kernel capacity")
        key=(m,torch.cuda.current_stream().cuda_stream,shared)
        if key in self.calls: return self.calls[key]
        probe_trace.note("expert_plan_begin",layer=getattr(self,"layer",-1),tokens=m)
        plan=fused_moe.plan_execution(experts=self.prepared,
            capacity=fused_moe.ExecutionCapacity(max_tokens=m,top_k=self.topk))
        fused_moe.prewarm(plan)
        scratch=shared_scratch(plan)
        buffers=execution_buffers(m,self.topk,shared)
        x,ids,weights,out=(buffers[name] for name in ("x","ids","weights","out"))
        binding=fused_moe.bind(plan,scratch=scratch,experts=self.prepared,a=x,
            topk_ids=ids,topk_weights=weights,output=out,input_scales_static=True)
        call={'plan':plan,'scratch':scratch,'x':x,'ids':ids,'weights':weights,'out':out,'binding':binding,'graph':None}
        self.calls[key]=call
        probe_trace.note("expert_plan_end",layer=getattr(self,"layer",-1),tokens=m)
        return call

    def execute(self,x,weights,mapped,shape):
        limit,_=execution_options()
        if x.shape[0]>limit:
            # Keep a whole prefill's expert group resident while reusing the
            # configured kernels and their fixed graph/scratch buffers.
            # Copy each result before the next replay overwrites its backing.
            output=torch.empty_like(x,dtype=torch.float32)
            topk=shape[1]
            for begin in range(0,x.shape[0],limit):
                end=min(begin+limit,x.shape[0])
                output[begin:end].copy_(self.execute(
                    x[begin:end],weights[begin:end],mapped[begin*topk:end*topk],
                    (end-begin,topk)))
            return output
        with step_profile.phase('expert_graph'):
            call=self.execution(x.shape[0])
            m=x.shape[0]
            call['x'][:m].copy_(x)
            call['ids'][:m].copy_(torch.tensor(mapped,dtype=torch.int32,device='cpu').reshape(shape))
            call['weights'][:m].copy_(weights)
            if m<call['x'].shape[0]:
                call['x'][m:].zero_()
                call['ids'][m:].fill_(-1)
                call['weights'][m:].zero_()
            if os.environ.get('DSV41_EXPERT_GRAPHS','1')=='1':
                if call['graph'] is None:
                    # Native prefetch readers synchronize their own CUDA
                    # stream. Quiesce them before a GLOBAL capture begins.
                    self.retire_prefetch(wait=True)
                    for cache in _layers.values(): cache.retire_prefetch(wait=True)
                    fused_moe.run(binding=call['binding'])
                    torch.cuda.synchronize()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        fused_moe.run(binding=call['binding'])
                    call['graph']=graph
                call['graph'].replay();_stats['graph_replays']+=1
            else:
                fused_moe.run(binding=call['binding'])
            return call['out'][:m]

    def run(self,x,weights,ids,on_load_start=None):
        stream=torch.cuda.current_stream()
        if self.last_stream is not None and self.last_stream!=stream.cuda_stream:
            stream.wait_event(self.last_event)
        result=self._run(x,weights,ids,on_load_start)
        self.last_event.record(stream)
        self.last_stream=stream.cuda_stream
        return result

    def _run(self,x,weights,ids,on_load_start=None):
        with step_profile.phase('slot_read'):
            needed=ids.reshape(-1).tolist()
            expert_io.trace_routes(self.layer,x.shape[0],self.count,needed,self.rank)
            if os.environ.get('DSV41_PREFETCH_TEST')=='1':
                import expert_prefetch
                expert_prefetch.observe(self,needed,x.shape[0])
            unique=set(needed)
            if len(unique)<=self.count:
                mapped=self.ensure(needed,on_load_start)
            else:
                mapped=None
        if mapped is not None:
            return self.execute(x,weights,mapped,ids.shape)
        import routed_pipeline
        if self.layer<40 and x.shape[0]>16 and routed_pipeline.enabled():
            return routed_pipeline.run(self,x,weights,ids,needed,on_load_start)
        # Consume already-resident experts before loading missing groups. Loading
        # an arbitrary first group can evict a resident expert needed by a later
        # group and reread it from SSD. The inputs/route IDs remain unchanged.
        frequency=Counter(needed[-64*self.topk:])
        resident=unique.intersection(self.used)
        ordered=sorted(unique-resident,key=lambda e:(frequency[e],e))
        groups=[sorted(resident,key=lambda e:(frequency[e],e))] if resident else []
        groups.extend(ordered[begin:begin+self.count] for begin in range(0,len(ordered),self.count))
        output=torch.zeros_like(x,dtype=torch.float32)
        for group in groups:
            with step_profile.phase('slot_read'):
                slots=self.ensure(group,on_load_start)
                remap=dict(zip(group,slots))
                mapped=[remap.get(e,-1) for e in needed]
            output.add_(self.execute(x,weights,mapped,ids.shape))
        return output


def apply(store,layer,x,weights,ids,limit,on_load_start=None):
    if x.shape[0]==0: return torch.zeros_like(x,dtype=torch.float32)
    if layer not in _layers: _layers[layer]=SlotLayer(store,layer,ids.shape[1])
    return _layers[layer].run(x,weights,ids,on_load_start)

def stats():
    for cache in _layers.values():
        for item in cache.prefetch_pending.values():
            if item['future'].done() and not item['future'].cancelled(): cache.account_prefetch(item)
    import routed_pipeline
    return _stats|routed_pipeline.stats()|{'prefetch_pending':sum(len(c.prefetch_pending) for c in _layers.values()),
        'execution_buffer_bytes':sum(t.numel()*t.element_size() for t in {id(c[k]):c[k] for cache in _layers.values() for c in cache.calls.values() for k in ('x','ids','weights','out')}.values()),
        'scratch_bytes':sum(a.numel() for a in _arenas.values()),'gpu_cache_bytes':sum(sum(t.numel()*t.element_size() for t in cache.parts) for cache in _layers.values())}
