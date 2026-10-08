"""Trial-only hooks: fail at the first non-finite eager module output."""
import torch


def install(model):
    def checker(name):
        def hook(module, args, kwargs, output):
            if torch.cuda.is_current_stream_capturing(): return
            batch=kwargs.get('forward_batch')
            if batch is None:
                batch=next((v for v in args if hasattr(v,'forward_mode')),None)
            if batch is None or batch.forward_mode.name != 'EXTEND': return
            values = output if isinstance(output, (tuple,list)) else (output,)
            for value in values:
                if not isinstance(value,torch.Tensor) or not value.is_floating_point(): continue
                if not torch.isfinite(value).all().item():
                    inputs = {f'arg_{i}':v.detach().cpu() for i,v in enumerate(args) if isinstance(v,torch.Tensor)}
                    inputs.update({k:v.detach().cpu() for k,v in kwargs.items() if isinstance(v,torch.Tensor)})
                    inputs['output'] = value.detach().cpu()
                    torch.save(inputs,'/trial-cache/first-nonfinite.pt')
                    raise RuntimeError(f'QAD_FIRST_NONFINITE module={name} class={type(module).__name__} shape={tuple(value.shape)}')
        return hook
    for name,module in model.named_modules():
        cls=type(module).__name__
        if any(part in cls for part in ('DecoderLayer','GatedDeltaNet','Attention','PLELayer','SparseMoeBlock')):
            module.register_forward_hook(checker(name),with_kwargs=True)
    print('QAD eager non-finite diagnostic hooks installed',flush=True)
