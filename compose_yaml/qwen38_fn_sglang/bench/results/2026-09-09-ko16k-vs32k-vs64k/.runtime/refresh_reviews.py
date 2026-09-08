import json,subprocess,sys
from pathlib import Path
out=Path(__file__).resolve().parent.parent
bench=out.parents[1]
args=[sys.executable,str(bench/'prepare_broad_review.py'),str(out),'--modes','ko64k','ko16k']
if not (out/'ko16k-benchmark.json').exists():args.append('--partial')
subprocess.run(args,check=True)
known=json.loads((out/'reusable-prose-reviews.json').read_text())
if (out/'prose-reviews.json').exists():
 known.update({r['review_id']:r for r in json.loads((out/'prose-reviews.json').read_text())})
packets=json.loads((out/'review-packets.json').read_text())
reviews=[known[r['review_id']] for r in packets if r['review_id'] in known]
pending=[r for r in packets if r['review_id'] not in known]
(out/'prose-reviews.json').write_text(json.dumps(reviews,ensure_ascii=False,indent=2)+'\n')
(out/'pending-reviews.json').write_text(json.dumps(pending,ensure_ascii=False,indent=2)+'\n')
print('Reviewed:',len(reviews),'pending:',len(pending))
print([(r['review_id'],r['task_id'],len(r['answer'])) for r in pending])
