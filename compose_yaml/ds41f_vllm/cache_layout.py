"""Validated per-layer allocation of the unchanged total expert-slot budget."""
import json,os
from pathlib import Path

_layouts={}


def load_layout(name,default):
    key=(name,default)
    if key not in _layouts:
        if not name:
            slots=[default]*40
        else:
            doc=json.loads((Path(__file__).resolve().parent/name).read_text())
            slots=doc['target_slots']
            if (len(slots)!=40 or any(type(n) is not int or not 6<=n<=384 for n in slots)
                    or sum(slots)>40*default):
                raise ValueError('Invalid or over-budget target expert allocation')
        _layouts[key]=tuple(slots)
    return _layouts[key]


def capacity(layer,default):
    if layer>=40:return default
    return load_layout(os.environ.get('DSV41_CACHE_LAYOUT',''),default)[layer]


if __name__=='__main__':
    import sys
    print(sum(load_layout(sys.argv[1],int(sys.argv[2]))))
