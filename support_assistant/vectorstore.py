import os
import glob
import chromadb
from chromadb.utils import embedding_functions

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

def init_vectorstore():
    # Use sentence-transformers all-MiniLM-L6-v2
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name="zepto_policies",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Ingest documents if collection is empty
    if collection.count() == 0:
        doc_files = sorted(glob.glob(os.path.join(DOCS_DIR, "doc_*.txt")))
        documents = []
        ids = []
        metadatas = []
        
        for file_path in doc_files:
            doc_id = os.path.basename(file_path).replace(".txt", "")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            documents.append(content)
            ids.append(doc_id)
            metadatas.append({"source": doc_id})
            
        if documents:
            collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
            print(f"Successfully ingested {len(documents)} documents into ChromaDB.")
            
    return collection

def query_vectorstore(query_text: str, top_k: int = 3):
    collection = init_vectorstore()
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k
    )
    
    retrieved_chunks = []
    if results and "documents" in results and results["documents"]:
        docs = results["documents"][0]
        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        distances = results.get("distances", [[]])[0]
        
        for doc, doc_id, meta, dist in zip(docs, ids, metadatas, distances):
            retrieved_chunks.append({
                "id": doc_id,
                "text": doc,
                "metadata": meta,
                "distance": dist
            })
            
    return retrieved_chunks

if __name__ == "__main__":
    init_vectorstore()
