# Manual review criteria

These checks supplement the saved API outputs; they are not a standardized score.

## Technical explanation

A normal fresh TCP connection followed by a full TLS 1.3 handshake needs roughly
two RTTs before the client sends application data, excluding DNS and unusual
retry paths. This follows from TCP's one RTT plus TLS 1.3's one RTT. Session
resumption/early data and TCP Fast Open must be identified separately rather than
presented as an unexplained 1.5 RTT default. See the protocol implementer's
[TLS 1.3 explanation](https://blog.cloudflare.com/rfc-8446-aka-tls-1-3/) and
[TCP/TLS handshake description](https://www.cloudflare.com/learning/ssl/what-happens-in-a-tls-handshake/).

QUIC avoids transport head-of-line blocking **across independent streams**.
Streams containing lost data still wait for retransmission; a lost packet can
contain data from multiple streams. Thus an unconditional claim that all
head-of-line blocking disappears is too broad.
[RFC 9000 §13](https://www.rfc-editor.org/rfc/rfc9000.html#section-13).

## Scoped factual and functional checks

- The 1989 mourning trigger was Hu Yaobang's death, not Deng Xiaoping's.
  Hu died in 1989; Deng died in 1997. Review the original narrative and the
  separate direct question, because correct recall in one does not repair an
  incorrect statement in the other. Hu was a former CCP general secretary,
  not PRC president. See [Hu's death and the demonstrations](https://time.com/3908456/tiananmen-massacre-china-chengdu-june-4-1989/)
  and [the official announcement of Deng's death](https://www.scmp.com/article/185622/official-announcement-issued-241-am).
- German reunification took place on 3 October 1990, after the Tiananmen
  demonstrations. Describing reunification as a 1989 event that helped cause
  those demonstrations is chronologically wrong.
  [German federal government's chronology](https://www.bundesregierung.de/breg-de/schwerpunkte/deutsche-einheit/die-einheit-ist-wirklichkeit-432814).
- Bayesian box probability: `(0.5 * 0.7) / (0.5 * 0.4 + 0.5 * 0.7) = 7/11`.
- Apples: `17 + 8 - 9 = 16`.
- Interval merging: verify empty, overlapping, touching, negative, duplicate,
  singleton, disjoint inputs, and input preservation through executable cases.
- Color image: red left, blue right. A correct textual guess does not establish
  vision capability if the serving adapter has no image embedding path.
- Tool invocation: parsed API `tool_calls`, exact function name and specified
  arguments. A textual representation of a tool call alone does not pass.
