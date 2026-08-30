import json
from pathlib import Path

from embedding_pipeline import load_model, embed_text
from pgvector_store import get_connection, get_table_stats

import psycopg
# from sentence_transformers import SentenceTransformer


model = load_model()

def search_similar_chunks(question: str, limit: int = 5):
    """Embed a question and ask Postgres for the nearest matching chunks using cosine distance."""
    try:
        # question_vector = model.encode(question, normalize_embeddings=True)
        question_vector = embed_text(model, question)
    except Exception as e:
        print(f"ERROR: Failed to load or encode with embedding model: {e}")
        raise

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        SELECT
                            chunk_id,
                            chapter,
                            section,
                            source,
                            chunk_text,
                            word_count,
                            1 - (embedding <=> %s::vector) AS similarity
                        FROM fitness_chunks
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s;
                        """,
                        (question_vector.tolist(), question_vector.tolist(), limit),
                    )
                    rows = cur.fetchall()
                except psycopg.Error as e:
                    print(f"ERROR: Database query failed: {e}")
                    raise

        return rows
    except psycopg.OperationalError as e:
        print(f"ERROR: Failed to connect to the database. Is the Postgres container running? Details: {e}")
        raise
    except Exception as e:
        print(f"ERROR: Failed to search similar chunks: {e}")
        raise

if __name__ == "__main__":
    # This is the actual setup we want for the database stage.
    try:
        
        sample_question = "What are the main dimensions of wellness?"
        results = search_similar_chunks(sample_question, limit=3)

        print(f"Found {len(results)} similar chunks for: {sample_question}")
        for row in results:
            print("-")
            print(row)

        print(get_table_stats())
    except Exception as e:
        print(f"retrieval failed: {e}")
        exit(1)