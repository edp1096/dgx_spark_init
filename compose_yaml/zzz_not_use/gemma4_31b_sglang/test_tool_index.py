"""Run inside the Gemma image: python3 /tmp/test_tool_index.py (no GPU)."""
import json
from sglang.srt.function_call.gemma4_detector import Gemma4Detector
from sglang.srt.entrypoints.openai.protocol import Tool

tools = [Tool(type='function', function={'name': n, 'parameters': {'type': 'object', 'properties': {}}})
         for n in ('unused', 'get_weather', 'get_stock_price')]
for names in [('get_weather',)*3, ('get_stock_price', 'get_weather', 'get_stock_price')]:
    text = ''.join('<|tool_call>call:' + name + '{value:' + str(i) + '}<tool_call|>'
                   for i, name in enumerate(names))
    for size in (0, 1, 7, len(text)):
        parser = Gemma4Detector()
        calls = []
        if size == 0:
            calls = parser.detect_and_parse(text, tools).calls
        else:
            for i in range(0, len(text), size):
                calls.extend(parser.parse_streaming_increment(text[i:i+size], tools).calls)
        assembled = {}
        for call in calls:
            item = assembled.setdefault(call.tool_index, {'name': None, 'args': ''})
            if call.name: item['name'] = call.name
            item['args'] += call.parameters or ''
        assert list(assembled) == [0, 1, 2], (size, assembled)
        assert [v['name'] for v in assembled.values()] == list(names)
        assert [json.loads(v['args']) for v in assembled.values()] == [{'value': i} for i in range(3)]
print('PASS: repeated/mixed calls, nonstream and stream chunk sizes 1/7/full')
