import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def ingestion_pipeline(pdf_path, db_storage_directory):

    #1. Document Loading
    print("Loading PDF document...")
    Loader = PyPDFLoader(pdf_path)
    row_documents = Loader.load()
    print(f"Successfully loaded {len(row_documents)} pages.")

    # 2. Text Chunking
    print("Chunking text into smaller pieces...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50
    )
    text_chunks = text_splitter.split_documents(row_documents)
    print(f"Split documents into {len(text_chunks)} chunks.")

    # 3. Text Embedding 
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Vector Storage
    print("Initializing local ChromaDB and saving vectors...")
    vector_db = Chroma.from_documents(
        documents=text_chunks,
        embedding=embedding_model,
        persist_directory=db_storage_directory
    )
    print("Vectors saved successfully.")

if __name__ == "__main__":

    SAMPLE_PDF = "cattlevetlook.pdf"
    LOCAL_DB_DIR = "./chroma_db"

    if os.path.exists(SAMPLE_PDF):
        ingestion_pipeline(SAMPLE_PDF, LOCAL_DB_DIR)
    else:
        print(f"Please place a sample PDF named '{SAMPLE_PDF}' in this folder first!")