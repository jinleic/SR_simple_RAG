# ollama_integration.py
import requests
import json

def query_ollama(prompt, system_prompt=""):
    """
    Query the local LLM via Ollama.
    Combines a system prompt with the user prompt and returns the LLM's response.
    """
    # Combine the system prompt with the user prompt if a system prompt is provided.
    combined_prompt = f"{system_prompt}\n{prompt}" if system_prompt else prompt

    # Revised payload using the structure from the working code snippet.
    payload = {
        "model": "qwq:latest",
        "prompt": combined_prompt,
        "stream": False
    }
    
    try:
        # Use the same endpoint and headers as in the working code example.
        response = requests.post(
            "http://localhost:11434/api/generate",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        response.raise_for_status()
        result = response.json().get("response", "")
        return result
    except Exception as e:
        return f"Error querying Ollama: {str(e)}"

# ----------------- Test Function ----------------- #
def test_query_ollama():
    test_prompt = "What is RAG?"
    response_text = query_ollama(
        test_prompt,
        system_prompt="You are an expert on retrieval-augmented generation."
    )
    print("Ollama Response:")
    print(response_text)
    assert len(response_text) > 0, "LLM returned an empty response."

if __name__ == "__main__":
    test_query_ollama()

