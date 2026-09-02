"""Live Avaal API → MongoDB sync (orders).

Replaces the manual JSON-file ingest: pulls the order list from the live Avaal
API on a schedule / on demand and upserts into the tenant Mongo collection,
re-embedding only records whose text changed.
"""
