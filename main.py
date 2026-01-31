from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. SETUP & CONFIG ---
app = FastAPI(title="Java Support Bot API")

# Initialize Embeddings (Must match Phase 3)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load existing Vector Store
vector_db = Chroma(
    persist_directory="./rag_vector_db",
    embedding_function=embeddings
)

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- 2. RAG LOGIC ---
# the system prompt (The "Guardrails")
system_prompt = (
    "You are an expert QnA assistant. Use the following pieces of retrieved "
    "context to answer the user's question. If the answer isn't in the context, "
    "say you don't know—don't try to make up an answer. Keep it concise.\n\n"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create the chains
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vector_db.as_retriever(), question_answer_chain)


# --- 3. API ENDPOINTS ---
class QueryRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_bot(request: QueryRequest):
    try:
        response = rag_chain.invoke({"input": request.question})
        return {
            "answer": response["answer"],
            "sources": [doc.metadata.get("source", "Unknown") for doc in response["context"]]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
