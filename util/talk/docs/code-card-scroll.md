# Streaming code-card scroll preservation

Streaming replaced the whole Markdown HTML, recreating code cards and losing
expanded state and `pre` scroll offsets. Moving an expanded card during a DOM
update could also temporarily shrink the message pane and clamp its scrollTop.

`web/src/lib/markdown-view.js` now reuses code cards when the same-language code
continues with an append-only update. It snapshots both code scroll axes and the
message-pane position before mutation, restores them synchronously, and retains
the existing expansion controls. Non-append replacements get fresh cards.
Both answer and reasoning Markdown use the action. HTML is still sanitized by
DOMPurify before rendering.

Validation: 89 frontend tests and 5 browser tests passed. The streaming regression
covers two code blocks, one expanded and one scrolled horizontally/vertically,
appended code, the surrounding reading position, and completion/save. Existing
completion-scroll and message-virtualization checks also pass.

The Linux ARM64 binary was rebuilt and applied. The running app's JavaScript
matches the tested asset (SHA256
`0accf6f8eb2ec55f4c2514b31607bb298d871856db1b8dd18b57aa285d73427f`).
Only SparkTalk was restarted; model serving settings were not changed.


## 2026-09-13: toggling while a chunk arrives

Reusing a card was not sufficient: moving it into a temporary fragment and then
back into the message detached the button during pointer down/up. Chromium
cancelled that click. A streaming regression reproduced the failure before the
fix by deliberately delivering a chunk between pointer down and pointer up.

The Markdown action now reconciles surrounding nodes in place and leaves a
retained code card connected. Its text grows without moving its controls.
The regression repeats expand/collapse three times while the response is still
streaming. Scroll preservation, completion/save and message virtualization
checks also pass.

Expanded long code cards also expose a bottom `↑ 접기` control. Collapsing there
returns to and focuses the header toggle. The footer is added when a streamed
short block crosses the long-code threshold, and stays connected with the card.
The streaming regression alternates top and bottom collapse while new chunks
arrive, then checks that the header remains visible and can expand again.
