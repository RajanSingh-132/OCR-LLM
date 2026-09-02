import uvicorn
import os
import sys
import asyncio

# Windows defaults to ProactorEventLoop, which psycopg's async mode does
# not support (needed for app/postgres_client.py). Force SelectorEventLoop
# on Windows before the app starts.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.fast_api_actions_session_rag import app

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8090))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=True,
    )