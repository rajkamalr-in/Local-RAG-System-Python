from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


def ask_rag_system(user_question, db_directory):
    # 1. Load the existing loacl vector DB
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(
        persist_directory=db_directory,
        embedding_function=embedding_model,
    )

    # 2. Convert DB into a retriver
    retriever = vector_db.as_retriever(
        search_kwargs={"k":3} # k:3 Means retrive the top 3 most relevant chuncks
    )

    # 3. Initialize local LLM (Called Ollama LLama 3)
    llm = OllamaLLM(model="llama3")

    # 4. Define the system prompt
    system_prompt = (
        "You are an expert technical assistant. Use the following pieces of retrieved "
        "context to answer the user's question. If you don't know the answer, say that "
        "you don't know—do not try to make up an answer. Keep the answer concise.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",system_prompt),
        ("human","{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever,question_answer_chain)

    print(f"\n Searching DB and generating answer for: '{user_question}'...\n")

    response = rag_chain.invoke({
        "input":user_question
    })

    return response

if __name__ == "__main__":
    LOCAL_DB_DIR = "./chroma_db"
    QUESTION = "Who is author of CattleVetLook?"
    
    result = ask_rag_system(QUESTION, LOCAL_DB_DIR)
    
    print("\n" + "="*30 + " FINAL LLM ANSWER " + "="*30)
    print(result.get("answer", "No answer found."))
    print("="*78 + "\n")