import json
import sys
import os

def test_api():
    # 1. Server Configuration
    server_port = 8090
    api_url = f"http://127.0.0.1:{server_port}/api/v1/orders/ask"
    
    # 2. Enter your test question here:
    test_question = "What is the status of Order ID 1062?"
    if len(sys.argv) > 1:
        test_question = " ".join(sys.argv[1:])

    payload = {
        "question": test_question,
        "collection_name": "orders"
    }

    print("=" * 60)
    print("           AIM RAG SERVICE - API TESTER            ")
    print("=" * 60)
    print(f"Target Endpoint : {api_url}")
    print(f"Test Question   : '{test_question}'")
    print("-" * 60)
    print("Sending request... Please wait...")

    # We use requests if available, otherwise fallback to Python's built-in urllib
    try:
        import requests
        try:
            response = requests.post(api_url, json=payload, timeout=45.0)
            status_code = response.status_code
            try:
                res_data = response.json()
                answer = res_data.get("answer", "No answer field returned.")
            except Exception:
                status_code = response.status_code
                answer = response.text
        except requests.exceptions.ConnectionError:
            print("\nError: Could not connect to the server.")
            print(f"Please make sure your FastAPI app is running on port {server_port}.")
            print("Run command: python main.py")
            return
    except ImportError:
        # Fallback to standard library urllib (zero dependencies)
        import urllib.request
        import urllib.error
        
        req = urllib.request.Request(
            api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=45.0) as response:
                status_code = response.status
                res_bytes = response.read()
                res_data = json.loads(res_bytes.decode("utf-8"))
                answer = res_data.get("answer", "No answer field returned.")
        except urllib.error.URLError as e:
            print("\nError: Could not connect to the server.")
            print(f"Please make sure your FastAPI app is running on port {server_port}.")
            print("Run command: python main.py")
            return
        except Exception as e:
            status_code = "Error"
            answer = str(e)

    print("-" * 60)
    print(f"Response Status : {status_code}")
    print("-" * 60)
    print("ANSWER FROM AI  :")
    print(answer)
    print("=" * 60)

if __name__ == "__main__":
    test_api()
