# RAG Support Bot

This project is a simple yet powerful Question-Answering (Q&A) bot that uses a Retrieval-Augmented Generation (RAG) pipeline to provide answers based on a local knowledge base.

## Features

- **FastAPI Backend**: A modern, fast (high-performance) web framework for building APIs.
- **RAG Pipeline**: Built with LangChain to retrieve relevant information and generate answers.
- **ChromaDB**: Used as a local vector store to save and query document embeddings.
- **Google Generative AI**: Powered by Google's `gemini-2.5-flash` model for generating answers.
- **HuggingFace Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` for creating text embeddings.

## Setup and Installation

1.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd rag-support-bot
    ```

2.  **Install Dependencies**:
    This project requires Python 3.8+. You can install the necessary packages using `pip`.
    ```bash
    pip install fastapi uvicorn "langchain[llms]" langchain-google-genai langchain-huggingface langchain-chroma python-dotenv
    ```
    *Note: A `requirements.txt` file is recommended for production setups.*

3.  **Set Up API Key**:
    -   Create a `.env` file in the root directory by copying the example:
        ```bash
        cp .env.example .env
        ```
        If `.env.example` doesn't exist, create a new file named `.env`.

    -   Add your Google API key to the `.env` file:
        ```
        # .env
        GOOGLE_API_KEY=your_google_api_key_here
        ```

4.  **Create the Knowledge Base**:
    The vector store in `rag_vector_db/` is pre-built. If you need to rebuild it or use your own data, you will need to run the `crawler.py` script (assuming it's configured to create the database).
    ```bash
    python crawler.py
    ```

## Running the Application

Once the setup is complete, you can start the API server using `uvicorn`.

```bash
uvicorn main:app --reload
```

The server will be running at `http://127.0.0.1:8000`.

## Usage (API)

You can send questions to the bot by making a `POST` request to the `/ask` endpoint.

**Example using `curl`**:
```bash
curl -X POST "http://127.0.0.1:8000/ask" \
-H "Content-Type: application/json" \
-d '{"question": "What is the difference between an abstract class and an interface in Java?"}'
```

**Example Response**:
```json
{
  "answer": "An abstract class can have both abstract and non-abstract methods, and a class can only extend one abstract class. An interface can only have abstract methods (in older Java versions), and a class can implement multiple interfaces. Abstract classes can also have instance variables, while interfaces cannot.",
  "sources": [
    "java_cheat_sheet_db/java_basics.txt"
  ]
}
```
