from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# 1. Scrape the content
def scrape_data():
    loader = WebBaseLoader("https://www.interviewbit.com/java-cheat-sheet/")
    docs = loader.load()

    # with open("content.txt", "w") as f:
    #     for doc in docs:
    #         f.write(doc.page_content.replace("\n", " "))

    for doc in docs:
        doc.page_content = doc.page_content.replace("\n", " ")

    # 2. Define the chunking strategy
    # We use 1000 characters with a 200 character overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True  # Helpful for referencing original source
    )

    # 3. Create the chunks
    all_splits = text_splitter.split_documents(docs)

    return all_splits

def generate_embeddings(all_splits):
    # 1. Embedding Model
    # This will download the model (approx 80MB) on the first run
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 2. the Vector Store
    # 'persist_directory' is where your database will live on your disk
    persist_directory = "./rag_vector_db"

    print("Generating embeddings and creating vector store...")
    vector_db = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    print(f"Vector store created and saved to {persist_directory}")

    # 3. Simple Retrieval Test
    query = "What is the difference between JDK and JRE?"
    docs = vector_db.similarity_search(query, k=3)

    print(f"\nTop Match Found:\n{docs[0].page_content[:300]}...")

def main():
    all_splits = scrape_data()
    generate_embeddings(all_splits)


if __name__ == '__main__':
    main()
