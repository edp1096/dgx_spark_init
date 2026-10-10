"""Keep quoted Qwen XML as text; pass actual call bodies to the existing parser.

The scanner owns only the text/call boundary. Parameter conversion and JSON
streaming remain in the installed Qwen detector. Reasoning is already separated
before SGLang calls this detector, so its fence state belongs to the answer only.
"""
import re

START = '<tool_call>'
FUNC = '<function='


class Guard:
    def __init__(self, channels=False):
        self.pending = ''
        self.line = []
        self.fence = None
        self.tool_end = None
        self.channels = channels

    def consume(self, text, update_fences=True):
        for part in text.splitlines(keepends=True):
            self.line.append(part)
            if part.endswith('\n'):
                line = ''.join(self.line).rstrip('\r\n')
                if not update_fences:
                    self.line = []
                    continue
                if self.fence:
                    char, width = self.fence
                    if re.fullmatch(r' {0,3}'+re.escape(char)+'{'+str(width)+r',}[ \t]*', line):
                        self.fence = None
                else:
                    match = re.match(r' {0,3}(`{3,}|~{3,})(.*)$', line)
                    if match and (match[1][0] != '`' or '`' not in match[2]):
                        self.fence = (match[1][0], len(match[1]))
                self.line = []

    def fenced(self):
        if self.fence:
            return True
        return bool(re.match(r' {0,3}(`{3,}|~{3,})', ''.join(self.line)))

    def feed(self, text, final=False):
        self.pending += text
        segments=[]
        def emit(kind, value):
            if not value:return
            if segments and segments[-1][0] == kind:
                segments[-1]=(kind,segments[-1][1]+value)
            else:segments.append((kind,value))
            self.consume(value, update_fences=kind=='text')
        while self.pending:
            value=self.pending
            if self.tool_end:
                pos=value.find(self.tool_end)
                if pos>=0:
                    end=pos+len(self.tool_end);emit('call',value[:end]);self.pending=value[end:];self.tool_end=None
                    continue
                keep=0
                if not final:
                    for width in range(1,min(len(value),len(self.tool_end)-1)+1):
                        if self.tool_end.startswith(value[-width:]):keep=width
                end=len(value)-keep
                emit('call',value[:end]);self.pending=value[end:]
                break
            if self.channels:
                boundary=next((token for token in ('<think>','</think>') if value.startswith(token)),None)
                if boundary:
                    emit('text',boundary);self.pending=value[len(boundary):]
                    self.line=[];self.fence=None
                    continue
                if not final and any(token.startswith(value) for token in ('<think>','</think>')):
                    break
            if value[0]!='<' or self.fenced():
                pos=value.find('<',1)
                end=pos if pos>=0 else len(value)
                # Preserve newline transitions before deciding a later marker's fence state.
                newline=value.find('\n')
                if newline>=0:end=min(end,newline+1)
                emit('text',value[:end]);self.pending=value[end:];continue
            if not final and any(token.startswith(value) for token in (START,FUNC)):
                break
            if value.startswith(START):
                rest=value[len(START):].lstrip()
                if not final and (not rest or FUNC.startswith(rest)):
                    break
                if rest.startswith(FUNC):
                    self.tool_end='</tool_call>'
                    continue
            elif value.startswith(FUNC) and not ''.join(self.line).strip():
                self.tool_end='</function>'
                continue
            emit('text','<');self.pending=value[1:]
        return segments


def install():
    from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
    from sglang.srt.function_call.core_types import StreamingParseResult
    if getattr(Qwen3CoderDetector,'_literal_guard_installed',False):return
    original_stream=Qwen3CoderDetector.parse_streaming_increment
    original_detect=Qwen3CoderDetector.detect_and_parse

    def streaming(self,text,tools):
        if not hasattr(self,'_literal_guard'):self._literal_guard=Guard()
        normal=[];calls=[]
        for kind,value in self._literal_guard.feed(text):
            if kind=='text':normal.append(value)
            else:
                r=original_stream(self,value,tools);normal.append(r.normal_text);calls+=r.calls
        return StreamingParseResult(normal_text=''.join(normal),calls=calls)

    def flush(self):
        if not hasattr(self,'_literal_guard'):return ''
        return ''.join(value for kind,value in self._literal_guard.feed('',final=True) if kind=='text')

    def finish(self,tools):
        return StreamingParseResult(normal_text=flush(self))

    def detect(self,text,tools):
        normal=[];calls=[]
        for kind,value in Guard().feed(text,final=True):
            if kind=='text':normal.append(value)
            else:
                r=original_detect(self,value,tools);normal.append(r.normal_text)
                for call in r.calls:
                    call.tool_index=len(calls);calls.append(call)
        return StreamingParseResult(normal_text=''.join(normal),calls=calls)

    Qwen3CoderDetector.parse_streaming_increment=streaming
    Qwen3CoderDetector.detect_and_parse=detect
    Qwen3CoderDetector.flush_guard=flush
    Qwen3CoderDetector.finish=finish
    Qwen3CoderDetector._literal_guard_installed=True
    install_reasoning_guard()


def install_reasoning_guard():
    from sglang.srt.parser.reasoning_parser import Qwen3Detector, StreamingParseResult
    if getattr(Qwen3Detector,'_literal_guard_installed',False):return
    original_stream=Qwen3Detector.parse_streaming_increment
    original_finish=Qwen3Detector.finish

    def process(self,segments):
        reasoning=[];normal=[]
        for kind,value in segments:
            if kind=='call' and self._in_reasoning:
                # Only a confirmed, unfenced function header closes reasoning implicitly.
                reasoning.append(self._buffer);self._buffer='';self._in_reasoning=False
                self._accumulated_reasoning='';normal.append(value)
            else:
                saved=self.tool_start_token
                try:
                    self.tool_start_token=None
                    result=original_stream(self,value)
                finally:self.tool_start_token=saved
                reasoning.append(result.reasoning_text);normal.append(result.normal_text)
        return StreamingParseResult(reasoning_text=''.join(reasoning),normal_text=''.join(normal))

    def streaming(self,text):
        if not hasattr(self,'_boundary_guard'):self._boundary_guard=Guard(channels=True)
        return process(self,self._boundary_guard.feed(text))

    def finish(self):
        result=process(self,self._boundary_guard.feed('',final=True)) if hasattr(self,'_boundary_guard') else StreamingParseResult()
        remaining=original_finish(self)
        return StreamingParseResult(reasoning_text=result.reasoning_text+remaining.reasoning_text,
                                    normal_text=result.normal_text+remaining.normal_text)

    def detect(self,text):
        fresh=Qwen3Detector(stream_reasoning=True,force_reasoning=self.force_reasoning,
                            continue_final_message=self.continue_final_message,previous_content=self.previous_content,
                            force_nonempty_content=self._force_nonempty_content)
        result=fresh.parse_streaming_increment(text);remaining=fresh.finish()
        return StreamingParseResult(reasoning_text=result.reasoning_text+remaining.reasoning_text,
                                    normal_text=result.normal_text+remaining.normal_text)

    Qwen3Detector.parse_streaming_increment=streaming
    Qwen3Detector.finish=finish
    Qwen3Detector.detect_and_parse=detect
    Qwen3Detector._literal_guard_installed=True
