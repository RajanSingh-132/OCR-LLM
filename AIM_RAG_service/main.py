import uvicorn
import os

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8090))

if __name__ == "__main__":
    uvicorn.run(
        "app.fast_api_actions_session_rag:app",
        host=HOST,
        port=PORT,
        reload=True,
    )
    