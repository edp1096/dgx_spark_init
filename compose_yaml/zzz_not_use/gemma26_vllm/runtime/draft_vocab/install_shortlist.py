"""Install optional draft-only vocabulary projection in pinned vLLM classes."""
import pathlib
root=pathlib.Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models')
for name in ['qwen3_5_mtp.py','gemma4_mtp.py']:
 p=root/name;s=p.read_text()
 marker='        # SPARKTALK_DRAFT_SHORTLIST\n'
 if marker in s:continue
 start=s.index('    def compute_logits(')
 body=s.index('    ) -> torch.Tensor | None:\n',start)+len('    ) -> torch.Tensor | None:\n')
 hook = "    def prepare_draft_lm_head(self, lm_head):\n        import os\n        if os.environ.get('DRAFT_SHORTLIST'):\n            from draft_shortlist import prepare\n            prepare(self, os.environ['DRAFT_SHORTLIST'])\n\n"
 s=s[:start]+hook+s[start:body]+marker+"        import os\n        if os.environ.get('DRAFT_SHORTLIST'):\n            from draft_shortlist import logits as shortlist_logits\n            return shortlist_logits(self, hidden_states)\n"+s[body:]
 p.write_text(s)
