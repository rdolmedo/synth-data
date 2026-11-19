# Response budgets and termination

The response budget covers the whole episode: reasoning, tool-call arguments, and
answer text. Tool observations do not spend that budget, but still consume context.
Valid tool calls remain allowed throughout the finishing allowance, including the
reserved finalization portion. Disabling further thinking does not disable tools.

The intermediate warning boundary and finalization boundary can interrupt a model
turn. Pi's pinned continuation patch can request another turn while budget remains.
Once the **total** budget is exhausted, the response proxy returns a local synthetic
`stop` response. It performs no further model inference or resampling. Harbor then
runs the verifier against the workspace as it stands. A complete valid tool call
generated at the boundary can finish executing before this local stop. An incomplete
tool call is never executed. Existing artifacts are not rolled back.

The synthetic response-cap marker is distinct from the context-cap marker and gets
zero SFT loss. A final model `length` response may cause one Pi continuation request
to the local proxy; that request gets the synthetic stop, not another model sample.

## Accounting before tool parsing

The Pi response proxy requests `return_token_ids` and uses the backend's `/detokenize`
endpoint. Both pinned vLLM 0.20.1 and 0.32 runtimes support these endpoints/fields.
The full raw output is counted before tool parsing. A parser can discard an
unfinished argument while returning a short, incomplete RPC labeled `tool_calls`;
that must not refund the thousands of tokens actually generated. Truncated tool
output is retained as non-executable text with a `length` stop while budget remains.
Complete valid tool calls are preserved, even when they finish exactly at the cap.

Missing or inconsistent raw accounting is an infrastructure error, not a model
answer. Once that error occurs, subsequent transport retries cannot perform more
paid inference in that episode. `/detokenize` gets one bounded retry first (2 tries
x 30 s). The guard's error 502 carries `X-Response-Budget-Error`, so the sticky
proxy passes it to Pi instead of holding it for `--hold` (#531: held, it hung the
session); the eval_q35 worker records the attempt as infrastructure
`response_budget_accounting` (same-seed retry).

`generation_accounting` records charged tokens, backend-native tokens, the raw
Snowball count, truncation, and accounting policy. `real_body_ids` describes the
retained canonical SFT message; its length can differ from the charged generation.
The two measurements must not be substituted for one another.

## Units

`PI_RESPONSE_BUDGET` keeps `token_counting: "snowball"` as its default. The Pi proxy
now counts the full raw output in that tokenizer, rather than only the surviving
parsed reply. Actual generation cropped at a boundary is still charged. With a
different backend tokenizer, a single backend call can cross a Snowball boundary;
the excess is recorded and charged, and cannot trigger inference after the total
is spent. A backend-native hard limit avoids this conversion issue.

Set `token_counting: "backend-native"` to count the backend's complete generated
token IDs for Pi, or full completion usage in the single-turn sync/async adapters.
Per-call native limits then enforce the episode budget in those same native units.
**Compute train/test quantiles in the selected tokenizer.** Do not reuse numerical
Snowball thresholds while describing them as Super-native thresholds.

Historical experiments and their frozen code/data are unchanged. New budget-policy
runs should record the accounting policy and units along with their thresholds.
