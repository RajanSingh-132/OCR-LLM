"""
"Rename" a Mongo database. Mongo has no rename-database command, so every
collection is copied to the new database with the same name, the same documents
(including _id) and the same indexes. Nothing inside the documents is changed.

  python scripts/rename_db.py --dry-run       # only show what would be copied
  python scripts/rename_db.py                 # copy, keep the old database
  python scripts/rename_db.py --drop-source   # copy, verify, then drop the old database

Safe to run again: a document already in the new database (same _id) is left as
it is. The old database is dropped only with --drop-source, and only if every
collection copied without a problem. MONGO_URI is read from .env.
"""
import argparse
import os
import sys
from typing import Dict, List, Tuple

from pymongo.errors import BulkWriteError, OperationFailure

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.mongo_client import get_mongo_client

# ===================== CONFIG — edit these =====================
SOURCE_DB = "AFMDEMO"
TARGET_DB = "AFMDEMOAI"
BATCH_SIZE = 1000
# ===============================================================

# Index options the server reports but does not accept (or need) on create.
_INDEX_SKIP_OPTIONS = {"v", "ns", "key", "background"}
_INDEX_CONFLICT_CODES = {85, 86}


def _insert_batch(target, batch: List[dict]) -> Tuple[int, int]:
    """Insert a batch; returns (inserted, already present). Other errors raise."""
    try:
        return len(target.insert_many(batch, ordered=False).inserted_ids), 0
    except BulkWriteError as exc:
        errors = exc.details.get("writeErrors", [])
        other = [e for e in errors if e.get("code") != 11000 or "_id_" not in e.get("errmsg", "")]
        if other or exc.details.get("writeConcernErrors"):
            raise
        return exc.details.get("nInserted", 0), len(errors)


def _copy_indexes(source, target) -> int:
    copied = 0
    for index in source.list_indexes():
        if index["name"] == "_id_":
            continue
        options = {k: v for k, v in index.items() if k not in _INDEX_SKIP_OPTIONS}
        try:
            target.create_index(list(index["key"].items()), **options)
        except OperationFailure as exc:
            if exc.code not in _INDEX_CONFLICT_CODES:
                raise
            print(f"  index {index['name']}: kept the one already in {TARGET_DB}")
            continue
        copied += 1
    return copied


def copy_collection(source_db, target_db, info: dict) -> Dict[str, int]:
    name = info["name"]
    source, target = source_db[name], target_db[name]

    if name not in target_db.list_collection_names():
        target_db.create_collection(name, **info.get("options", {}))

    inserted = existing = 0
    batch: List[dict] = []
    for doc in source.find({}, batch_size=BATCH_SIZE):
        batch.append(doc)
        if len(batch) >= BATCH_SIZE:
            done, skipped = _insert_batch(target, batch)
            inserted, existing, batch = inserted + done, existing + skipped, []
    if batch:
        done, skipped = _insert_batch(target, batch)
        inserted, existing = inserted + done, existing + skipped

    return {
        "inserted": inserted,
        "existing": existing,
        "indexes": _copy_indexes(source, target),
        "source_total": source.count_documents({}),
        "target_total": target.count_documents({}),
    }


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=f"Copy {SOURCE_DB} to {TARGET_DB}")
    parser.add_argument("--dry-run", action="store_true", help="only show what would be copied")
    parser.add_argument("--drop-source", action="store_true", help=f"drop {SOURCE_DB} after a clean copy")
    args = parser.parse_args(argv)

    client = get_mongo_client()
    client.admin.command("ping")

    if SOURCE_DB not in client.list_database_names():
        print(f"Database {SOURCE_DB} not found. Nothing to do.")
        return 1

    source_db, target_db = client[SOURCE_DB], client[TARGET_DB]
    infos = [i for i in source_db.list_collections() if not i["name"].startswith("system.")]
    plain = [i for i in infos if i.get("type", "collection") == "collection"]
    skipped = [i["name"] for i in infos if i.get("type", "collection") != "collection"]
    target_names = set(target_db.list_collection_names())

    print(f"\n=== {SOURCE_DB} -> {TARGET_DB}{' (dry run)' if args.dry_run else ''} ===")
    if skipped:
        print(f"Not copied (views etc.): {skipped}")

    if args.dry_run:
        for info in plain:
            name = info["name"]
            in_target = target_db[name].count_documents({}) if name in target_names else 0
            print(f"  {name}: {source_db[name].count_documents({})} docs, "
                  f"{in_target} already in {TARGET_DB}")
        print()
        return 0

    problems: List[str] = []
    for info in plain:
        name = info["name"]
        stats = copy_collection(source_db, target_db, info)
        print(f"  {name}: {stats['inserted']} copied, {stats['existing']} already there, "
              f"{stats['indexes']} indexes, total {stats['source_total']} -> {stats['target_total']}")
        if stats["inserted"] + stats["existing"] != stats["source_total"]:
            problems.append(name)

    if problems:
        print(f"\nCount mismatch in {problems}. {SOURCE_DB} was NOT dropped; run again.\n")
        return 1

    if not args.drop_source:
        print(f"\nDone. {SOURCE_DB} was kept. Use --drop-source to remove it.\n")
    elif skipped:
        print(f"\nDone. {SOURCE_DB} was NOT dropped because {skipped} were not copied.\n")
    else:
        client.drop_database(SOURCE_DB)
        print(f"\nDone. {SOURCE_DB} was dropped.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
