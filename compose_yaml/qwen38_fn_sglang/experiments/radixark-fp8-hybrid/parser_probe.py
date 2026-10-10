"""Regression cases for quoted Qwen tool XML, using the installed SGLang parser."""
import json
from pathlib import Path
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector

TOOL = Tool.model_validate({'type':'function','function':{'name':'move_room','parameters':{
    'type':'object','properties':{'folder':{'type':'string'}},'required':['folder']}}})
CALL = '<tool_call>\n<function=move_room>\n<parameter=folder>뉴스</parameter>\n</function>\n</tool_call>'
CASES = [
    ('real', CALL, True),
    ('real_after_prose', '이동하겠다.\n'+CALL, True),
    ('literal_marker', '문자열 <tool_call>은 도구 호출의 시작 표식이다. 설명 완료.', False),
    ('fenced_xml', '```xml\n'+CALL+'\n```\n설명 완료.', False),
    ('tilde_fence', '~~~xml\n'+CALL+'\n~~~\n설명 완료.', False),
    ('nested_fence', '````markdown\n```xml\n'+CALL+'\n```\n````\n설명 완료.', False),
    ('real_after_fence', '```xml\n<tool_call> 예시\n```\n'+CALL, True),
    ('inline_function', '문자열 <function=move_room>은 함수 헤더 예시다. 설명 완료.', False),
    ('unclosed_fence', '```xml\n'+CALL, False),
    ('trailing_marker', '문자열 표식: <tool_call>', False),
    ('trailing_partial', '문자열 표식: <tool_ca', False),
    ('real_after_parameter_fence', CALL.replace('뉴스','```python\nprint(1)')+'\n'+CALL, True),
]


def probe():
    results=[]
    for name,text,expected in CASES:
        detector=Qwen3CoderDetector();r=detector.detect_and_parse(text,[TOOL])
        passed=bool(r.calls)==expected and (expected or r.normal_text==text)
        results.append(dict(name=name,mode='one_shot',pass_=passed,normal=r.normal_text,
                            calls=[x.model_dump() for x in r.calls]))
        for width in [1,7,100000]:
            detector=Qwen3CoderDetector();normal='';calls=[]
            for offset in range(0,len(text),width):
                r=detector.parse_streaming_increment(text[offset:offset+width],[TOOL])
                normal+=r.normal_text;calls+=r.calls
            ending=detector.finish([TOOL]);normal+=ending.normal_text;calls+=ending.calls
            passed=bool(calls)==expected and (expected or normal==text)
            results.append(dict(name=name,mode=f'stream_{width}',pass_=passed,normal=normal,
                                calls=[x.model_dump() for x in calls]))
    return results


if __name__=='__main__':
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='patched':
        import parser_guard
        parser_guard.install()
    result=probe()
    print(json.dumps(dict(passed=sum(x['pass_'] for x in result),total=len(result),results=result),ensure_ascii=False,indent=2))
    if len(sys.argv)>1 and sys.argv[1]=='patched':assert all(x['pass_'] for x in result)
