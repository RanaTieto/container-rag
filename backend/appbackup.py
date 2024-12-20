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

    # Pass retrieved documents to LLM for answer generation
    try:
        context = "\n".join([doc[1] for doc in results])
        llm_response = requests.post(f"{LLM_URL}/api/generate", json={"prompt": f"Context: {context}\nQuery: {query}", "model": "llama3.2:1b"})
        llm_response.raise_for_status()  # This will raise an exception if the response code is not 200
        logging.debug(f"LLM response content: {llm_response.text}")
        response_json = llm_response.json()

        # Collect the chunks of the response
        response_chunks = []
        for line in llm_response.iter_lines(decode_unicode=True):
            if line.strip():  # Ignore empty lines
                response_chunks.append(line)
        
        # Combine and parse the final JSON
        full_response = "".join(response_chunks)
        response_json = json.loads(full_response)

        answer = response_json.get("response", "")
        return {"answer": answer}
        #return {"answer": response_json["answer"]}
    except requests.exceptions.RequestException as e:
        logging.error(f"Error contacting LLM: {e}")
        raise HTTPException(status_code=500, detail="Again Failed to generate answer")
    except ValueError as e:
        logging.error(f"Error parsing JSON response: {e}")
        raise HTTPException(status_code=500, detail="Invalid response format from LLM")
    
    #return {"answer": llm_response.json()["answer"]}
