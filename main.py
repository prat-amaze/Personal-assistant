from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from graph import graph
import time
from collections import defaultdict

app = FastAPI()

# ---- Rate Limiter (per IP) ----
RATE_LIMIT = 3
WINDOW = 60  # seconds

requests_log = defaultdict(list)

def is_rate_limited(ip: str):
    now = time.time()

    # keep only recent requests
    requests_log[ip] = [t for t in requests_log[ip] if t < WINDOW]

    if len(requests_log[ip]) >= RATE_LIMIT:
        return True

    requests_log[ip].append(now)
    return False


# ---- Chat Endpoint ----
@app.get("/chat")
def chat(q: str, request: Request):
    ip = request.headers.get("x-forwarded-for", request.client.host)

    if is_rate_limited(ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Max 3 requests per minute."
        )

    result = graph.invoke({"query": q})

    return {
        "query": q,
        "category": result.get("category"),
        "answer": result.get("answer")
    }


# ---- Serve frontend ----
app.mount("/", StaticFiles(directory="static", html=True), name="static")