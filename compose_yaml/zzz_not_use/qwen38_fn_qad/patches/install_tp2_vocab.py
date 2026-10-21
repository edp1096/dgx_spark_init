from pathlib import Path
import ast
p=Path('/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py')
s=p.read_text()
a='''                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]'''
b='''                self.hot_token_id = self.hot_token_id.to(head.device)
                if get_parallel().tp_size == 2:
                    from sglang.srt.speculative.tp_vocab import make_tp_shortlist
                    head = make_tp_shortlist(head, target_lm_head, self.hot_token_id)
                else:
                    head = head.clone()
                    head.data = head.data[self.hot_token_id]'''
assert s.count(a)==1
s=s.replace(a,b);ast.parse(s);p.write_text(s)
