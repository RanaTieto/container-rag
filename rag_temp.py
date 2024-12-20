import requests

try:
    response = requests.post("http://embeddings:8001/generate", json={"text": "test"})
    print(response.status_code, response.text)
except Exception as e:
    print(f"Error: {e}")
