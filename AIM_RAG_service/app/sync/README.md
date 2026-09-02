# Live Avaal API → MongoDB sync (orders)

Replaces the manual `scripts/ingest/order_ingest.py` file-dump ingest. Pulls the
order list from the live Avaal API and **upserts** into the tenant's Mongo
collection, re-embedding only records whose text changed. The read path (query
planner / analytics / RAG) is unchanged.

## Pieces

| Module | Role |
|---|---|
| `payload.py` | `unwrap_order_payload()` — turn any API/file envelope into `list[dict]` |
| `documents.py` | `build_page_content` / `content_hash` / `build_order_document` |
| `avaal_client.py` | `AvaalClient` — POST `/api/Order/listorder`, page iterator, retry |
| `sync_state.py` | `_sync_state` (cursor + run history) and `_sync_lock` collections |
| `order_sync.py` | `sync_orders(corporate_id, mode="incremental"|"full", dry_run=False)` |
| `scheduler.py` | optional background thread: incremental every N min + nightly full |

## Trigger

- **Manual / external:** `POST /api/v1/orders/sync` with header
  `X-Sync-Token: <AVAAL_SYNC_TOKEN>` and body `{"corporate_id": "...", "mode": "incremental"}`.
- **Automatic:** set `AVAAL_SYNC_ENABLED=1`; the API starts a daemon thread on
  startup that runs incremental syncs on an interval and one full + reconcile per
  day.

## `.env` keys

```
AVAAL_API_BASE_URL=http://173.209.153.108:5000
AVAAL_API_ORDER_PATH=/api/Order/listorder
AVAAL_API_TIMEOUT_SEC=60
AVAAL_API_MAX_RETRIES=4
AVAAL_API_USE_MODIFIED_FILTER=0     # 1 once datetype="MODIFIEDON" is confirmed server-side
AVAAL_SYNC_USERCODE=USR00001
AVAAL_SYNC_USERNAME=Rahul Agrawal
AVAAL_SYNC_PAGESIZE=1000
AVAAL_SYNC_TOKEN=<random secret for the /sync endpoint>

# scheduler (only if AVAAL_SYNC_ENABLED=1)
AVAAL_SYNC_ENABLED=0
AVAAL_SYNC_INTERVAL_MIN=15
AVAAL_SYNC_FULL_HOUR=2              # UTC hour for the daily full+reconcile run
AVAAL_SYNC_TENANTS=AFN01514,AFN01619
```

`MONGO_URI` and `DB_NAME` must already be set (same as the read path).

## Rollout

1. `sync_orders("AFN01514", mode="full", dry_run=True)` — check counts, no writes.
2. `sync_orders("AFN01514", mode="full")` — first real load; sets the cursor.
3. Hit `/api/v1/orders/sync` with `mode=incremental` a few times; confirm changed
   orders appear and re-runs are no-ops.
4. Set `AVAAL_SYNC_ENABLED=1` + `AVAAL_SYNC_TENANTS` to turn on the scheduler.
5. Copy the pattern for invoices / trips.

## Notes / open items

- `AVAAL_API_USE_MODIFIED_FILTER=0` by default — incremental relies on paging
  `ModifiedOn DESC` and stopping at the stored cursor. Turn the flag on only after
  confirming the API honours `datetype="MODIFIEDON"` + `fromdate`.
- Deleted/cancelled orders: flagged `is_stale=true` from the record's own
  `isdeleted`/`isactive`, plus the nightly full run marks any `ordernumber` not
  returned. Nothing is hard-deleted.
- Identity key is `orderid`. Legacy docs from the old ingest keyed only by
  `ordernumber` may need a one-time cleanup if `orderid` was absent.
