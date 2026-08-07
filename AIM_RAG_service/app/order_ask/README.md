# Order Ask (`/api/v1/orders/ask`)

Advanced Avaal OrderBot: conversation + accurate lists + tool Q&A.

```
app/order_ask/
  checkpoint.py     # terminal [CHECKPOINT] logs
  memory.py         # session turns (Mongo avaal_chat_sessions)
  entities.py       # extract filters / order tokens
  tools.py          # get_order, search_orders, calc, compare, RAG
  intent.py
  rag_retrieval.py
  calculation_engine.py
  prompts.py
  rag_engine.py
```

## Request

```json
{
  "question": "list confirmed orders",
  "session_id": null
}
```

Reuse `session_id` from the response for follow-ups like `"uska tax?"`.

## Terminal

Watch server stdout for lines like:

```
[CHECKPOINT] ASK_START — new request | question=...
[CHECKPOINT] INTENT — local fast-path | intent=list_filter
[CHECKPOINT] TOOL_RUN — start search_orders
[CHECKPOINT] ASK_END — complete
```
