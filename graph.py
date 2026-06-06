from langgraph.graph import StateGraph
from typing import TypedDict, List
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ---- State ----
class State(TypedDict):
    query: str
    category: str
    context: List[str]
    answer: str

# ---- Setup ----
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# ---- Valid categories ----
VALID = [
    "Education",
    "Work Experience",
    "Projects",
    "Technical Skills",
    "Contact Information",
    "Positions of Responsibility",
    "Professional Summary", 
    "roles"
]

def normalize(category: str):
    for v in VALID:
        if v.lower() in category.lower():
            return v
    return "Professional Summary"


# ---- Nodes ----

# 1. Classifier
def classify(state):
    prompt = f"""
Classify the query into one of these categories:
{", ".join(VALID)}

Query: {state["query"]}

Return ONLY the category name.
"""

    res = llm.invoke(prompt)
    category = normalize(res.content.strip())

    return {"category": category}


# 2. Retriever
def retrieve(state):
    docs = db.similarity_search(
        state["query"],
        k=4,
        filter={"type": state["category"]}
    )

    # fallback if nothing found
    if not docs:
        docs = db.similarity_search(state["query"], k=4)

    return {"context": [d.page_content for d in docs]}


# 3. Answer
def generate(state):
    context = "\n\n".join(state["context"])

    prompt = f"""
Answer the question about me using the context below.

Context:
{context}

Question: {state["query"]}

If the answer is not in the context, say "I don't have that information".
Always answer in first person perspective
"""

    res = llm.invoke(prompt)

    return {"answer": res.content}


# ---- Graph ----
builder = StateGraph(State)

builder.add_node("classify", classify)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)

builder.set_entry_point("classify")
builder.add_edge("classify", "retrieve")
builder.add_edge("retrieve", "generate")

graph = builder.compile()

if __name__ == "__main__":
    queries = [
        # "What is your CGPA?",
        # "Tell me about your projects",
        # "What skills do you have?",
        # "What is your email?",
        "What did you do at Chubb?"
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        result = graph.invoke({"query": q})
        print(f"Category: {result.get('category')}")
        print(f"Answer: {result.get('answer')}")
        print("-" * 50)