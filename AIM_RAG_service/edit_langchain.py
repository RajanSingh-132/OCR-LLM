import sys

file_path = 'c:/Users/singh/Desktop/AIM_RAG_service2/AIM_RAG_service/app/lan_chain_rag_semantic_parent.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove MongoVectorStore block
start_idx = content.find('# ---------------- MONGO VECTOR STORE ----------------')
end_idx = content.find('# ---------------- METADATA ----------------')

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + content[end_idx:]

# 2. Add import
import_stmt = 'from app.rag_retrieval import get_vectorstore'
if import_stmt in content:
    content = content.replace(import_stmt, import_stmt + '\nfrom app.prompt import DYNAMIC_EXTRACTION_PROMPT')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print('Success')
