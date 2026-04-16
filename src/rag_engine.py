import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from src.prompts import STRICT_PROMPT, FLEXIBLE_PROMPT

os.makedirs("data", exist_ok=True)
CHROMA_PATH = os.path.join("data", "chroma_db")

def setup_rag_chain(strict_mode=True, provider="Groq (Llama 3.1 8B)"):
    """
    Sets up the connections to the DB, the LLM, and the Memory.
    Returns a 'Chain' that we can ask questions to.
    """
    print("🧠 Waking up the AI and connecting to the database...")

    # --- 1. CONNECT TO VECTOR DB ---
    # We must use the exact same embedding model we used to save the data!
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
    
    # We turn the database into a "Retriever". 
    # search_kwargs={"k": 6} means "fetch the top 6 most relevant chunks".
    retriever = db.as_retriever(search_kwargs={"k": 6})

    # 2. THE ROUTER: Pick the LLM(brain) based on the dropdown
    if provider == "Groq (Llama 3.1 8B)":
        llm = ChatGroq(
            model="llama-3.1-8b-instant", 
            temperature=0.1
        )
    elif provider == "Google (Gemini 2.5 Flash)":
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.1
        )
    else:
        # Fallback just in case
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1
        )

    # --- 2. SETUP THE Google LLM ---
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-flash",
    #     temperature=0.1,
    # )

    # # --- 2.1 SETUP THE Groq LLM ---
    # llm = ChatGroq(
    #     model="llama-3.1-8b-instant",
    #     temperature=0.1,
    # )

    # --- 3. SETUP CONVERSATION MEMORY ---
    # This keeps track of the chat history so the bot remembers context.
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )

    # --- 4. Dynamically select the prompt ---
    selected_prompt = STRICT_PROMPT if strict_mode else FLEXIBLE_PROMPT

    # --- 5. CREATE THE PIPELINE (CHAIN) ---
    # It takes the query, searches the retriever, injects the memory, and queries the LLM as per prompt.
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": selected_prompt}
    )
    
    return qa_chain