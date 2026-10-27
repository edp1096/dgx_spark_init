"""Transcript checks are evidence about words, not a voice-identity/emotion score."""
import difflib
import re
import unicodedata


def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC',text).casefold() if c.isalnum())


def coverage(expected, transcript):
    a,b=normalize(expected),normalize(transcript)
    row=list(range(len(b)+1))
    for i,c in enumerate(a,1):
        out=[i]
        for j,d in enumerate(b,1):out.append(min(out[-1]+1,row[j]+1,row[j-1]+int(c!=d)))
        row=out
    cer=row[-1]/max(1,len(a));matched=set()
    for block in difflib.SequenceMatcher(None,a,b,autojunk=False).get_matching_blocks():matched.update(range(block.a,block.a+block.size))
    clauses=[];offset=0
    for piece in re.split(r'[.!?。！？]+\s*|\n+',expected):
        part=normalize(piece)
        if not part:continue
        fraction=sum(i in matched for i in range(offset,offset+len(part)))/len(part)
        clauses.append({'text':piece.strip(),'coverage':fraction,'characters':len(part)});offset+=len(part)
    missing=any(x['characters']>=4 and x['coverage']<.5 for x in clauses)
    return {'expected':expected,'transcript':transcript,'character_error_rate':cer,'exact':a==b,
            'clauses':clauses,'passed':bool(a) and cer<=.25 and not missing,
            'interpretation':'ASR comparison; discrepancies may be recognition errors, not proven omissions.'}
