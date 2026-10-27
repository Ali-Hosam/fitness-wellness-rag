import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

token = os.getenv("HF_ACCESS_TOKEN")

if not token:
    raise ValueError("HF_ACCESS_TOKEN was not found in the environment.")

client = InferenceClient(
    provider="deepinfra",
    api_key=token,
)

model_id = "Qwen/Qwen3-Embedding-0.6B"

text = "Hugging Face makes deploying embedding models incredibly easy."

print("Generating embedding...")

embedding = client.feature_extraction(
    text,
    model=model_id,
)

print("Embedding generated successfully!")
print("Embedding dimension:", len(embedding[0]))
print("First 5 values:", embedding[0:5])