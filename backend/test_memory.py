from app.memory import save_memory, search_memories

save_memory(
    "user1",
    "I am learning Python and artificial intelligence."
)

save_memory(
    "user1",
    "I enjoy building machine learning projects."
)

results = search_memories(
    "user1",
    "What programming language am I learning?"
)

for result in results:
    print(result)