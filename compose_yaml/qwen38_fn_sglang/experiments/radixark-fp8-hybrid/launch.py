"""Install experiment-only hooks in the parent and SGLang child processes."""
import os
import runpy
import sys

if __name__ == '__main__':
    sys.path.insert(0, '/experiment')
    import hybrid_dispatch
    hybrid_dispatch.install()
    runpy.run_path('/opt/radixark/launch.py', run_name='__main__')
