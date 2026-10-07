
import json
import math
import sqlite3
from ollama import Client

DB_PATH = "memory.db"
EMBEDDING_MODEL = "nomic-embed-text"

client = Client(host="http://localhost:11434")


def init_db():
    connection = sqlite3.connect(DB_PATH)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def create_embedding(text: str):
    response = client.embed(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response["embeddings"][0]


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def save_memory(user_id: str, content: str):
    embedding = create_embedding(content)

    connection = sqlite3.connect(DB_PATH)

    connection.execute(
        """
        INSERT INTO memories (user_id, content, embedding)
        VALUES (?, ?, ?)
        """,
        (
            user_id,
            content,
            json.dumps(embedding)
        )
    )

    connection.commit()
    connection.close()


def search_memories(
    user_id: str,
    query: str,
    limit: int = 3
):
    query_embedding = create_embedding(query)

    connection = sqlite3.connect(DB_PATH)

    rows = connection.execute(
        """
        SELECT content, embedding
        FROM memories
        WHERE user_id = ?
        """,
        (user_id,)
    ).fetchall()

    connection.close()

    results = []

    for content, embedding_json in rows:
        embedding = json.loads(embedding_json)

        similarity = cosine_similarity(
            query_embedding,
            embedding
        )

        results.append(
            {
                "content": content,
                "similarity": similarity
            }
        )

    results.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    return results[:limit]


init_db()