import { finishTool, startTool } from './tool-trace.js';

export function createStreamHandlers(message, publish) {
  return {
    reasoning(delta) {
      message.activity = 'reasoning';
      message.reasoning_content += delta;
      publish();
    },
    delta(delta) {
      message.activity = 'answer';
      message.content += delta;
      publish();
    },
    toolStart(data) {
      startTool(message, data);
      publish();
    },
    toolResult(data) {
      finishTool(message, data);
      publish();
    },
  };
}
