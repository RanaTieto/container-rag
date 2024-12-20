import requests
import psycopg2
from psycopg2.extras import Json
import random

# Configuration
DATABASE_URL = "postgresql://user:password@localhost:5432/vector_db"
BACKEND_URL = "http://localhost:8000/query"
HEADERS = {"Content-Type": "application/json"}

# Generate random 768-dimensional embeddings
def generate_random_embedding(dim=768):
    return [random.random() for _ in range(dim)]

# Sample Data with 768-dimensional embeddings
documents = [
    {"content": "Paris is the capital of France.", "embedding": generate_random_embedding()},
    {"content": "Berlin is the capital of Germany.", "embedding": generate_random_embedding()},
]

query_text = "What is the capital of France?"

# Functions
def insert_documents():
    """Insert sample documents into the PostgreSQL database."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        print("Connected to the database.")

        # Insert documents
        for doc in documents:
            cursor.execute(
                """
                INSERT INTO documents (content, embedding)
                VALUES (%s, %s);
                """,
                (doc["content"], Json(doc["embedding"])),
            )
        conn.commit()
        print("Inserted sample documents into the database.")
    except Exception as e:
        print(f"Error inserting documents: {e}")
    finally:
        cursor.close()
        conn.close()


def test_backend_query():
    """Test the backend service for query handling."""
    try:
        payload = {"query": query_text}
        response = requests.post(BACKEND_URL, json=payload, headers=HEADERS)
        if response.status_code == 200:
            print("Backend Response:")
            print(response.json())
        else:
            print(f"Error from backend: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error testing backend: {e}")


# Main Execution
if __name__ == "__main__":
    print("Step 1: Insert Documents into Database")
    insert_documents()

    print("\nStep 2: Test Backend Query")
    test_backend_query()
