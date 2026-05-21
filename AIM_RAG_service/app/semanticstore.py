import os
import datetime
import logging
from app.mongo_client import get_mongo_collection
from app.embedding_client import get_models

logger = logging.getLogger("api")

# Absolute path to getorderlist.txt
GETORDERLIST_PATH = r"C:\Users\singh\Desktop\AIM_RAG_service2\AIM_RAG_service\getorderlist.txt"
if not os.path.exists(GETORDERLIST_PATH):
    GETORDERLIST_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "getorderlist.txt"
    )

def store_user_question(question: str, embeddings) -> bool:
    """
    Dynamically generates a 1024-dimensional embedding for the user's question
    and stores it in the MongoDB collection under 'user_questions' namespace.
    """
    try:
        if not question or not question.strip():
            logger.warning("Empty question provided; skipping database storage.")
            return False
            
        logger.info(f"Dynamically generating 1024D embedding for question: '{question}'")
        
        # Get embedding from Bedrock client
        question_embedding = embeddings.embed_query(question)
        question_embedding = [float(x) for x in question_embedding]
        
        # Log embedding dimension size to verify
        logger.info(f"Successfully generated embedding of dimension: {len(question_embedding)}")
        
        collection = get_mongo_collection()
        doc = {
            "namespace": "user_questions",
            "page_content": question,
            "embedding": question_embedding,
            "metadata": {
                "created_at": datetime.datetime.utcnow().isoformat(),
                "type": "user_question",
                "embedding_dimensions": len(question_embedding)
            }
        }
        
        # Insert document dynamically
        result = collection.insert_one(doc)
        logger.info(f"Successfully stored user question in database with ID: {result.inserted_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to store user question in DB: {str(e)}", exc_info=True)
        return False
