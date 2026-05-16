# import os
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma

# def run_ingestion_pipeline(pdf_path, db_storage_directory):
#     # 1. DOCUMENT LOADING
#     print("🔄 Loading PDF document...")
#     loader = PyPDFLoader(pdf_path)
#     raw_documents = loader.load()
#     print(f"✅ Successfully loaded {len(raw_documents)} pages.")

#     # 2. TEXT CHUNKING
#     print("🔄 Chunking text into smaller pieces...")
#     # We use 500 characters per chunk with a 50-character overlap
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500, 
#         chunk_overlap=50
#     )
#     text_chunks = text_splitter.split_documents(raw_documents)
#     print(f"✅ Split document into {len(text_chunks)} chunks.")

#     # 3. TEXT EMBEDDING (Free Open-Source Model)
#     print("🔄 Loading embedding model (all-MiniLM-L6-v2)...")
#     # This downloads the model to your local machine on the first run
#     embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#     # 4. VECTOR STORAGE
#     print("🔄 Initializing local ChromaDB and saving vectors...")
#     # Chroma will automatically compute embeddings and store them on your disk
#     vector_db = Chroma.from_documents(
#         documents=text_chunks,
#         embedding=embedding_model,
#         persist_directory=db_storage_directory
#     )
#     print(f"🚀 Ingestion Complete! Vector DB saved locally at: '{db_storage_directory}'")

# if __name__ == "__main__":
#     # Example Usage:
#     # Put any sample PDF (like a project report or policy) in your directory
#     SAMPLE_PDF = "sample_policy.pdf" 
#     LOCAL_DB_DIR = "./chroma_db"
    
#     # Check if sample file exists before running
#     if os.path.exists(SAMPLE_PDF):
#         run_ingestion_pipeline(SAMPLE_PDF, LOCAL_DB_DIR)
#     else:
#         print(f"❌ Please place a sample PDF named '{SAMPLE_PDF}' in this folder first!")

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def ask_rag_system(user_question, db_directory):
    # 1. LOAD THE EXISTING LOCAL VECTOR DB
    # We MUST use the exact same embedding model we used during ingestion!
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_db = Chroma(
        persist_directory=db_directory, 
        embedding_function=embedding_model
    )
    
    # 2. CONVERT DB INTO A RETRIEVER
    # search_kwargs={"k": 3} tells it to fetch the top 3 most relevant chunks
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 3. INITIALIZE LOCAL LLM (Ollama Llama 3)
    llm = OllamaLLM(model="llama3")

    # 4. DEFINE THE SYSTEM PROMPT (Instructing the LLM how to behave)
    system_prompt = (
        "You are an expert technical assistant. Use the following pieces of retrieved "
        "context to answer the user's question. If you don't know the answer, say that "
        "you don't know—do not try to make up an answer. Keep the answer concise.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 5. TIE IT ALL TOGETHER INTO A RAG CHAIN
    # This chain handles stuffing the retrieved documents into the prompt
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # This final chain handles taking the user input, querying the retriever, 
    # and passing the result to the question_answer_chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # 6. RUN THE QUERY
    print(f"\n🔍 Searching DB and generating answer for: '{user_question}'...\n")
    response = rag_chain.invoke({"input": user_question})
    
    return response

if __name__ == "__main__":
    LOCAL_DB_DIR = "./chroma_db"
    
    # Change this question based on whatever PDF you ingested!
    QUESTION = "What is the main policy regarding remote work mentioned in the document?"
    
    result = ask_rag_system(QUESTION, LOCAL_DB_DIR)
    
    print("============= LLM ANSWER =============")
    print(result["answer"])
    print("======================================")