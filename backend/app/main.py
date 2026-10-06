from fastapi import FastAPI

app = FastAPI(
    title="Self-Learning AI Agent",
    description="AI assistant with persistent memory",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"message": "AI Agent API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}