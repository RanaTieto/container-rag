from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import psycopg2
from psycopg2.extras import execute_values
import logging
import json

# Setup basic logging
logging.basicConfig(level=logging.DEBUG)

# Initialize the FastAPI app
app = FastAPI()

DATABASE_URL = "postgresql://user:password@database:5432/vector_db"
#LLM_URL = "http://llm:5000"
LLM_URL = "http://llm:11434"
EMBEDDING_URL = "http://embeddings:8000"

# Database connection
def get_db():
    return psycopg2.connect(DATABASE_URL)

# Models
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query(request: QueryRequest):
    query = request.query

    # Generate embeddings for query
    try:
        embedding_response = requests.post(f"{EMBEDDING_URL}/generate", json={"text": query})
        embedding_response.raise_for_status()  # This will raise an exception if the response code is not 200
        query_embedding = embedding_response.json().get("embedding")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embeddings")

    # Query database for similar documents
    import numpy as np
    query_embedding = np.array(query_embedding).flatten().tolist()

    try:
        conn = get_db()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, content FROM documents
                ORDER BY embedding <-> %s::vector
                LIMIT 5;
            """, (query_embedding,))
            results = cursor.fetchall()
        conn.close()
    except Exception as e:
        logging.error(f"Error querying database: {e}")
        raise HTTPException(status_code=500, detail="Failed to query database")

    # Function to handle streaming response from LLM
    def handle_llm_streaming_response(url, payload):
        response = requests.post(url, json=payload, stream=True)
        if response.status_code == 200:
            full_response = ""  # To accumulate streamed responses
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        # Parse each fragment and extract 'response' key
                        fragment = json.loads(line)
                        if 'response' in fragment:
                            full_response += fragment['response']
                    except json.JSONDecodeError as e:
                        logging.error(f"Error decoding JSON fragment: {line}, error: {e}")
            return full_response
        else:
            logging.error(f"Request failed with status code {response.status_code}")
            raise HTTPException(status_code=500, detail="Failed to generate answer")

    # Pass retrieved documents to LLM for answer generation
    try:
        context = "\n".join([doc[1] for doc in results])
        llm_payload = {"prompt": f"Context: {context}\nQuery: {query}", "model": "llama3.2:1b"}
        answer = handle_llm_streaming_response(f"{LLM_URL}/api/generate", llm_payload)
        return {"answer": answer}
    except HTTPException as e:
        raise e  # Already logged and formatted
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error occurred")

    #return {"answer": llm_response.json()["answer"]}
