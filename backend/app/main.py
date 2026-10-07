from fastapi import FastAPI
from pydantic import BaseModel
from ollama import chat

from .memory import search_memories, save_memory


app = FastAPI(
    title="Self-Learning AI Agent",
    description="AI assistant with persistent long-term memory",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    user_id: str
    message: str


@app.get("/")
def root():
    return {
        "message": "AI Agent API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/chat")
def chat_with_agent(request: ChatRequest):

    memories = search_memories(
        request.user_id,
        request.message,
        limit=3
    )

    relevant_memories = "\n".join(
        f"- {memory['content']}"
        for memory in memories
        if memory["similarity"] > 0.40
    )

    if not relevant_memories:
        relevant_memories = "No relevant memories found."

    system_prompt = f"""
You are a helpful AI assistant with long-term memory.

Use the following memories about the user when they
are relevant to the current conversation.

USER MEMORIES:
{relevant_memories}

Rules:
- Use memories only when relevant.
- Do not invent memories.
- If the current user message conflicts with an old memory,
  prioritize the current message.
- Answer naturally and helpfully.
"""

    response = chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": request.message
            }
        ]
    )

    answer = response.message.content

    save_memory(
        request.user_id,
        request.message
    )

    return {
        "response": answer,
        "memories_used": [
            memory["content"]
            for memory in memories
            if memory["similarity"] > 0.40
        ]
    }