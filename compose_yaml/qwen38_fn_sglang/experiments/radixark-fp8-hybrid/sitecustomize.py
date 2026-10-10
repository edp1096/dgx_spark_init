# Python multiprocessing spawn imports this before model construction.
from hybrid_dispatch import install
install()
import os
if os.environ.get('RADIXARK_LITERAL_GUARD') == '1':
    import parser_guard
    parser_guard.install()
if os.environ.get('RADIXARK_FAST_LOADER') == '1':
    import loader_hooks
    loader_hooks.install()
