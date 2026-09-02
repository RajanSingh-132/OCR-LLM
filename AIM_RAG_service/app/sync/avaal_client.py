"""HTTP client for the live Avaal order-list API.

    POST {AVAAL_API_BASE_URL}{AVAAL_API_ORDER_PATH}
    headers: content-type/accept json + `corporateid: <CID>`   (no auth token)
    body:    {"Filter": { ...mostly-blank filter..., "pageno", "pagesize",
                          "sortcolumn": "ModifiedOn", "sortorder": "DESC" }}

The list response already contains full order objects, so no per-order detail
call is needed.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Iterator, List, Optional

import requests
from dotenv import load_dotenv

from app.sync.payload import unwrap_order_payload

load_dotenv(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
        ".env",
    )
)

logger = logging.getLogger("sync.avaal_client")

BASE_URL = os.environ.get("AVAAL_API_BASE_URL", "http://173.209.153.108:5000").rstrip("/")
ORDER_PATH = os.environ.get("AVAAL_API_ORDER_PATH", "/api/Order/listorder")
SYNC_USERCODE = os.environ.get("AVAAL_SYNC_USERCODE", "USR00001")
SYNC_USERNAME = os.environ.get("AVAAL_SYNC_USERNAME", "")
DEFAULT_PAGE_SIZE = int(os.environ.get("AVAAL_SYNC_PAGESIZE", "1000"))
REQUEST_TIMEOUT = int(os.environ.get("AVAAL_API_TIMEOUT_SEC", "60"))
# Send a server-side ModifiedOn>=cursor filter. OFF by default: not yet confirmed
# the API honours datetype="MODIFIEDON"; order_sync's client-side cutoff (sort
# ModifiedOn DESC, stop at the cursor) is the reliable path regardless.
USE_MODIFIED_FILTER = os.environ.get(
    "AVAAL_API_USE_MODIFIED_FILTER", "0"
).strip().lower() not in ("", "0", "false", "no", "off")
PAGE_PAUSE_SEC = float(os.environ.get("AVAAL_API_PAGE_PAUSE_SEC", "0.2"))
MAX_RETRIES = int(os.environ.get("AVAAL_API_MAX_RETRIES", "4"))
# Hard stop so a bad envelope / server never spins forever.
MAX_PAGES = int(os.environ.get("AVAAL_API_MAX_PAGES", "500"))

# The full Filter object from the real request — every field blank/neutral so the
# API validator is satisfied; only paging + sort are meaningful for a sync.
_BASE_FILTER: Dict[str, Any] = {
    "dataviewtype": "D", "companycode": "", "ordernumber": "", "customercode": "",
    "salesmancode": "", "customerordernumber": "", "orderstatus": "",
    "shipmenttype": "", "pickuplocation": "", "pickupcity": "", "deliverycity": "",
    "isdate": False, "datetype": "MODIFIEDON", "fromdate": "", "todate": "",
    "pickupstatecode": "", "deliverystatecode": "", "status": "", "currencycode": "",
    "csa": "", "hazmat": "", "overdimension": "", "searchvalue": "",
    "orderformtype": "", "equipmenttype": "", "usertype": "A", "username": "",
    "pageno": 1, "pagesize": DEFAULT_PAGE_SIZE,
    "sortcolumn": "ModifiedOn", "sortorder": "DESC",
    "accountingstatus": "", "statuscondition": "", "pickupcountrycode": "",
    "deliverycountrycode": "", "usercode": "", "orderoutid": -1, "carriercode": "",
    "tripnumber": "", "pickuprefnum": "", "pendingaccessorial": False,
    "advancesearchwhere": "",
}


class AvaalApiError(RuntimeError):
    pass


class AvaalClient:
    def __init__(
        self,
        *,
        base_url: str = BASE_URL,
        order_path: str = ORDER_PATH,
        usercode: str = SYNC_USERCODE,
        username: str = SYNC_USERNAME,
        page_size: int = DEFAULT_PAGE_SIZE,
        session: Optional[requests.Session] = None,
    ):
        self.url = f"{base_url}{order_path}"
        self.usercode = usercode
        self.username = username
        self.page_size = page_size
        self._session = session or requests.Session()

    def _build_body(
        self,
        *,
        page: int,
        modified_since: Optional[str],
        overrides: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        f = dict(_BASE_FILTER)
        f["usercode"] = self.usercode
        f["username"] = self.username
        f["pageno"] = int(page)
        f["pagesize"] = int(self.page_size)
        if modified_since and USE_MODIFIED_FILTER:
            # Optional server-side incremental filter (see USE_MODIFIED_FILTER).
            f["isdate"] = True
            f["datetype"] = "MODIFIEDON"
            f["fromdate"] = modified_since
        if overrides:
            f.update(overrides)
        return {"Filter": f}

    def fetch_page(
        self,
        corporate_id: str,
        *,
        page: int = 1,
        modified_since: Optional[str] = None,
        filter_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[dict]:
        body = self._build_body(
            page=page, modified_since=modified_since, overrides=filter_overrides
        )
        headers = {
            "content-type": "application/json",
            "accept": "application/json",
            "corporateid": corporate_id,
        }
        last_exc: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self._session.post(
                    self.url, json=body, headers=headers, timeout=REQUEST_TIMEOUT
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise AvaalApiError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                return unwrap_order_payload(resp.json())
            except (requests.RequestException, AvaalApiError, ValueError) as exc:
                last_exc = exc
                wait = min(30, 2 ** attempt)
                logger.warning(
                    "Avaal list page %s attempt %s/%s failed (%s) — retry in %ss",
                    page, attempt, MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)
        raise AvaalApiError(
            f"Avaal list failed for {corporate_id} page {page}: {last_exc}"
        )

    def iter_pages(
        self,
        corporate_id: str,
        *,
        modified_since: Optional[str] = None,
        filter_overrides: Optional[Dict[str, Any]] = None,
    ) -> Iterator[List[dict]]:
        """Yield record batches page by page; stop on a short/empty page."""
        for page in range(1, MAX_PAGES + 1):
            records = self.fetch_page(
                corporate_id,
                page=page,
                modified_since=modified_since,
                filter_overrides=filter_overrides,
            )
            if not records:
                return
            yield records
            if len(records) < self.page_size:
                return
            if PAGE_PAUSE_SEC:
                time.sleep(PAGE_PAUSE_SEC)
        logger.warning(
            "Avaal list for %s hit MAX_PAGES=%s — stopping", corporate_id, MAX_PAGES
        )
