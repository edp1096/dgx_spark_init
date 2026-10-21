import os

if os.environ.get('SPARKTALK_GEMMA_ASSISTANT_EXL3') == '1':
    try:
        from sglang_gemma_exl3 import install
        install()
    except Exception:
        import traceback
        traceback.print_exc()
        os._exit(1)
