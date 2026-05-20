import uvicorn
import os

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