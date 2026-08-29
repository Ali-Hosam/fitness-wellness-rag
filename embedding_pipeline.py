from pathlib import Path
from sentence_transformers import SentenceTransformer
from dataclasses import asdict, dataclass
from chunking import Chunk

# These paths keep the embedding stage connected to the output of chunking.py.
CHUNKS_PATH = Path(__file__).with_name("fitness_wellness_chunks.jsonl")
EMBEDDINGS_PATH = Path(__file__).with_name("fitness_wellness_embeddings.jsonl")

# This fixed model must also embed user questions later during retrieval.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@dataclass
class Embedded_Chunk:
    chunk_id: int
    chapter: str | None
    section: str | None
    text: str
    word_count: int
    embedding_vector: list[float]  # Store the embedding vector as a list for JSON serialization


def load_model() -> SentenceTransformer:
    """Load the chosen embedding model when the pipeline actually needs it."""
    return SentenceTransformer(MODEL_NAME)


def embed_text(model: SentenceTransformer, text: str):
    """Convert one piece of text into the model's NumPy embedding vector."""
    # NumPy output is convenient for validation, saving, and database insertion.
    return model.encode(text, normalize_embeddings=True)  # Normalize for cosine similarity

#function to embed one test chunk and validate its metadata(ID, chapter, word count) and vector dimensions and length
def embed_chunk(model: SentenceTransformer, chunk: Chunk):
    """Convert one chunk of text into the model's NumPy embedding vector."""
    text = chunk.text
    vector = embed_text(model, text)
    # Validate metadata
    assert chunk.chunk_id is not None
    assert chunk.chapter is not None
    assert chunk.word_count > 0
    # Validate vector dimensions and length
    assert len(vector) == 384, f"Expected vector length of 384, got {len(vector)}"
    return vector

#embedding test chunk from the chuunks file
# def embed_chunk_from_file(model: SentenceTransformer, chunk_id: int):
#     """Load a chunk from the JSONL file and embed it."""
#     import json
#     #the "r" is for read mode "w" is for write mode and "a" is for append mode
#     with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
#         for line in f:
#             chunk_data = json.loads(line)
#             if chunk_data["chunk_id"] == chunk_id:
#                 #the ** operator unpacks the dictionary into keyword arguments for the Chunk constructor
#                 chunk = Chunk(**chunk_data)
#                 return embed_chunk(model, chunk)
#     raise ValueError(f"Chunk with ID {chunk_id} not found in {CHUNKS_PATH}")

#embedding all chunks and saving in another file alongside all metadata
def embed_all_chunks(model: SentenceTransformer):
    """Embed all chunks from the JSONL file and save them to another JSONL file."""
    import json
    with open(CHUNKS_PATH, "r", encoding="utf-8") as input_file, open(EMBEDDINGS_PATH, "w", encoding="utf-8") as output_file:
        for line in input_file:
            chunk_data = json.loads(line)
            chunk = Chunk(**chunk_data)
            vector = embed_chunk(model, chunk)
            # Save the chunk metadata along with its embedding vector
            output_data =Embedded_Chunk(
                chunk_id=chunk.chunk_id,
                chapter=chunk.chapter,
                section=chunk.section,
                text=chunk.text,
                word_count=chunk.word_count,
                embedding_vector=vector.tolist()  # Convert NumPy array to list for JSON serialization
            )
            output_file.write(json.dumps(asdict(output_data), ensure_ascii=False) + "\n")

if __name__ == "__main__":
    embedding_model = load_model()
    # test_vector = embed_text(embedding_model, "one test sentence")
    # print(len(test_vector))

    # chunk = Chunk(
    #     chunk_id=2,
    #     chapter="Healthy Behaviors",
    #     section="Dimensions of Wellness",
    #     text="As most college students do, you have probably set goals...",
    #     word_count=163
    # )

    # test_vector = embed_test_chunk(embedding_model, chunk)
    # embedded_chunk_vector = embed_chunk_from_file(embedding_model, 2)
    # print(f"Embedded chunk vector length: {len(embedded_chunk_vector)}")

    #embedding all chunks and saving them to a new file
    embed_all_chunks(embedding_model)
    print(f"Embedded all chunks and saved to {EMBEDDINGS_PATH}")