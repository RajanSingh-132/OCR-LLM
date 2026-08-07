# Avaal static orders

Source used for ingest:
- `AIM_RAG_service/orderdata.txt` (905 records)

Mongo target:
- database: `chatbot_db`
- collection: `Avaal_db`
- namespace: `avaal_orders`

Code package:
- `app/order_ask/` (config, ingest, calculation_engine, rag_engine, prompts)

Re-ingest:
```bash
python -m scripts.ingest_avaal_orders
```
