"""Local trial API checks; these are functional probes, not quality benchmarks."""
import argparse
import json
import pathlib
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    base = 'http://127.0.0.1:8016/v1'
    results = []

    def run(name, messages, **options):
        body = dict(model='qwen38-qad-sglang-trial', messages=messages,
                    temperature=0, max_tokens=1024, stream=False)
        body.update(options)
        request = urllib.request.Request(
            base + '/chat/completions',
            data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'}, method='POST')
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
        results.append(dict(name=name, seconds=time.monotonic()-started,
                            request=body, response=result))
        # Preserve received evidence even if a semantic assertion fails.
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        choice = result['choices'][0]
        assert choice['finish_reason'] != 'length', (name, 'truncated response')
        return choice['message']

    message = run('plain_text', [{'role': 'user', 'content':
        '17 더하기 8의 값을 숫자만 답하세요.'}],
        chat_template_kwargs={'enable_thinking': False})
    assert (message.get('content') or '').strip() == '25', message

    message = run('reasoning', [{'role': 'user', 'content':
        'A box contains 12 red balls and 5 blue balls. How many balls total? '
        'Think briefly, then give only the number in your final answer.'}],
        chat_template_kwargs={'enable_thinking': True, 'reasoning_effort': 'low'})
    assert (message.get('content') or '').strip() == '17', message
    assert message.get('reasoning_content'), ('missing parsed reasoning', message)

    message = run('tool_call', [{'role': 'user', 'content':
        'Call get_weather for Seoul now. Use the tool; do not guess the weather.'}],
        chat_template_kwargs={'enable_thinking': False}, tools=[{
            'type': 'function', 'function': {
                'name': 'get_weather', 'description': 'Get current weather for a city.',
                'parameters': {'type': 'object', 'properties': {
                    'city': {'type': 'string'}}, 'required': ['city'],
                    'additionalProperties': False}}}], tool_choice='auto')
    calls = message.get('tool_calls') or []
    assert len(calls) == 1, message
    function = calls[0]['function']
    assert function['name'] == 'get_weather', message
    assert json.loads(function['arguments']) == {'city': 'Seoul'}, message
    print('Text, parsed reasoning and parsed tool-call probes passed')


if __name__ == '__main__':
    main()
