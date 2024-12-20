import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()
logger.info("Initializing embedding model...")
embedding_model = pipeline("feature-extraction", model="bert-base-uncased")
logger.info("Embedding model loaded successfully.")

# Define a schema for the request body
class EmbeddingRequest(BaseModel):
    text: str

@app.post("/generate")
def generate_embedding(request: EmbeddingRequest):
    try:
        # logger.info(f"Received text for embedding: {request.text}")
        # embedding = embedding_model(request.text)
        # logger.info(f"Generated embedding: {embedding[0][:5]}... (truncated for logging)")
        # return {"embedding": embedding[0]}

        raw_embedding = embedding_model(request.text)
        # Reduce dimensions (average across tokens)
        reduced_embedding = np.mean(raw_embedding[0], axis=0).tolist()
        logger.info(f"Generated embedding (truncated): {reduced_embedding[:5]}...")
        return {"embedding": reduced_embedding}
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Test embedding generation
if __name__ == "__main__":
    test_text = "What is the capital of France?"
    logger.info(f"Testing embedding generation with text: '{test_text}'")
    try:
        raw_embedding = embedding_model(test_text)
        reduced_embedding = np.mean(raw_embedding[0], axis=0).tolist()
        logger.info(f"Test embedding generated successfully: {reduced_embedding[:5]}... (truncated for display)")
        logger.info(f"Embedding size: {len(reduced_embedding)}")
    except Exception as e:
        logger.error(f"Error during test embedding generation: {str(e)}")

