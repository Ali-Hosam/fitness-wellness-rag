import json
from pathlib import Path

import psycopg
from dotenv import load_dotenv
import os

# This file is the bridge between our saved embeddings and Postgres.
# It creates the table, loads the JSONL embedding file, and runs a cosine-distance search.
# In plain English: it puts the meaning-numbers into a database so we can ask, "find the chunks that match this question?"

load_dotenv()  # Load environment variables from .env file
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "myproject_db"
DB_USER = "myuser"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDINGS_FILE = Path(__file__).with_name("fitness_wellness_embeddings.jsonl")


def get_connection():
    """Open a connection to the local Postgres database running in Docker."""
    return psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=os.environ["DB_PASSWORD"],  # Use the password from the .env file
    )


def create_table():
    """Create the vector table that will store each chunk and its embedding."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS fitness_chunks (
                    chunk_id INTEGER PRIMARY KEY,
                    chapter TEXT,
                    section TEXT,
                    source TEXT,
                    chunk_text TEXT,
                    word_count INTEGER,
                    embedding VECTOR(384)
                );
                """
            )
            conn.commit()

def create_indexes():
    """Create an index on the embedding column for fast similarity search."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_fitness_chunks_embedding ON fitness_chunks USING hnsw (embedding vector_cosine_ops);")
            conn.commit()

def get_table_stats():
    """Return the number of chunks in the database."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM fitness_chunks;")
            count = cur.fetchone()[0]
    return count

#jsonl_path can be string or path object, it will be converted to Path object if it is string
def insert_embeddings_from_jsonl(jsonl_path: str | Path):
    """Read all chunk records, convert the saved embedding list to a vector, and insert them."""
    #normalize the input for path even if it was string
    jsonl_path = Path(jsonl_path)
    count = 0
    
    # ERROR HANDLING: Check if the JSONL file exists before attempting to open it
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Embeddings file not found at: {jsonl_path.absolute()}")
    
    if not jsonl_path.is_file():
        raise ValueError(f"Expected a file but got a directory: {jsonl_path.absolute()}")

    try:
        with get_connection() as conn:
            #the cursor is used to execute the SQL commands
            with conn.cursor() as cur:
                with jsonl_path.open("r", encoding="utf-8") as input_file:
                    line_number = 0
                    for line in input_file:
                        line_number += 1
                        if not line.strip():
                            continue

                        try:
                            #json.loads parse the json into python dictionary
                            record = json.loads(line)
                        except json.JSONDecodeError as e:
                            raise ValueError(f"Invalid JSON at line {line_number}: {e}")

                        # The JSON metadata file contains the chunk data and the embedding vector.
                        # The source was not always stored in the earlier JSONL, so we set it explicitly.
                        # The ON CONFLICT ... DO UPDATE clause is used to update the existing record if the chunk_id already exists
                        # EXCLUDED is a special table that contains the values proposed for insertion, allowing us to update the existing record with the new values.
                        try:
                            cur.execute(
                                """
                                INSERT INTO fitness_chunks (
                                    chunk_id,
                                    chapter,
                                    section,
                                    source,
                                    chunk_text,
                                    word_count,
                                    embedding
                                )
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (chunk_id) DO UPDATE SET
                                    chapter = EXCLUDED.chapter,
                                    section = EXCLUDED.section,
                                    source = EXCLUDED.source,
                                    chunk_text = EXCLUDED.chunk_text,
                                    word_count = EXCLUDED.word_count,
                                    embedding = EXCLUDED.embedding
                                """,
                                (
                                    record["chunk_id"],
                                    record.get("chapter"),
                                    record.get("section"),
                                    record.get("source", "fitness_wellness_corpus_cleaned.txt"),
                                    record["text"],
                                    record["word_count"],
                                    record["embedding_vector"],
                                ),
                            )
                            count += 1
                        except psycopg.Error as e:
                            raise RuntimeError(f"Database error at line {line_number} (chunk_id {record.get('chunk_id')}): {e}")

            #commit all the inserts in one transaction
            conn.commit()
            print(f"Successfully inserted {count} embeddings from {jsonl_path}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        raise
    except psycopg.OperationalError as e:
        print(f"ERROR: Failed to connect to the database. Is the Postgres container running? Details: {e}")
        raise
    except Exception as e:
        print(f"ERROR: Failed to insert embeddings: {e}")
        raise




if __name__ == "__main__":
    # This is the actual setup we want for the database stage.
    try:
        create_table()
        insert_embeddings_from_jsonl(EMBEDDINGS_FILE)
        create_indexes()

        print(get_table_stats())
    except Exception as e:
        print(f"setup failed: {e}")
        exit(1)
