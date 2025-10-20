import os
from dotenv import load_dotenv
from openai import OpenAI

def get_api_key():
    load_dotenv() 
    return os.getenv("OPENAI_KEY")

def run_experiment():
    """Basic template for querying the openai api."""

    client = OpenAI(
    api_key=get_api_key()
    )

    response = client.responses.create(
        model="gpt-5",
        input="write a haiku about ai",
        store=False
    )

    print(response.output_text)

if __name__ == "__main__":
    run_experiment()
    