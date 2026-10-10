"""Check literal markers through both reasoning and function-call parsers."""
import json
import sys
if len(sys.argv)>1 and sys.argv[1]=='patched':
    import parser_guard
    parser_guard.install()
from sglang.srt.parser.reasoning_parser import Qwen3Detector
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from parser_probe import TOOL,CALL

cases=[
    ('reasoning_prose','문자열 <tool_call>은 표식이다.', '완료',False),
    ('reasoning_xml','```xml\n'+CALL+'\n```','완료',False),
    ('reasoning_tilde','~~~xml\n'+CALL+'\n~~~','완료',False),
    ('open_reasoning_fence','```xml\n예시',CALL,True),
    ('final_fenced','생각 중','```xml\n'+CALL+'\n```\n완료',False),
    ('real_after_think','생각 중',CALL,True),
    ('implicit_real','생각 중\n',CALL,True),
]
results=[]
for name,thought,answer,expected in cases:
    raw='<think>'+thought+('' if name=='implicit_real' else '</think>')+answer
    for stream_reasoning in [True,False]:
        for width in [0,1,7,100000]:
            reason=Qwen3Detector(stream_reasoning=stream_reasoning)
            function=Qwen3CoderDetector();reasoning='';normal='';calls=[]
            if width==0:
                r=reason.detect_and_parse(raw);reasoning=r.reasoning_text
                t=function.detect_and_parse(r.normal_text,[TOOL]);normal=t.normal_text;calls=t.calls
            else:
                for offset in range(0,len(raw),width):
                    r=reason.parse_streaming_increment(raw[offset:offset+width]);reasoning+=r.reasoning_text
                    t=function.parse_streaming_increment(r.normal_text,[TOOL]);normal+=t.normal_text;calls+=t.calls
                r=reason.finish();reasoning+=r.reasoning_text
                t=function.parse_streaming_increment(r.normal_text,[TOOL]);normal+=t.normal_text;calls+=t.calls
                t=function.finish([TOOL]);normal+=t.normal_text;calls+=t.calls
            passed=bool(calls)==expected and reasoning==thought and (expected or normal==answer)
            results.append(dict(name=name,width=width,stream_reasoning=stream_reasoning,pass_=passed,reasoning=reasoning,normal=normal,calls=[x.model_dump() for x in calls]))
print(json.dumps(dict(passed=sum(x['pass_'] for x in results),total=len(results),results=results),ensure_ascii=False,indent=2))
if len(sys.argv)>1 and sys.argv[1]=='patched':assert all(x['pass_'] for x in results)
