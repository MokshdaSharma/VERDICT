import os
from google import genai

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY (or Gemini API key)")
        return
        
    client = genai.Client(api_key=api_key)
    
    print("Available Models:")
    try:
        models = client.models.list()
        for m in models:
            print(f"- {m.name}")
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    main()
