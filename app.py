import os
import json
import shutil
import tempfile
import time
from dotenv import load_dotenv
import streamlit as st
from src.data_engine import process_document
from src.rag_engine import setup_rag_chain

load_dotenv()
os.makedirs("data", exist_ok=True)
HISTORY_FILE = os.path.join("data", "chat_history.json")

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title="Varta RAG AI",
    page_icon="🤖",
    layout="centered"
)
st.title("🤖 Varta RAG AI")
st.markdown("Upload your secure documents and chat with our AI.")
st.divider()

# 1. Create a cached wrapper function
@st.cache_resource(show_spinner="Booting AI Engine into RAM...")
def get_cached_chain(session_id, provider):
    return setup_rag_chain(
        strict_mode=st.session_state.get("strict_mode", True),
        provider=provider
    )

def stream_generator(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.04)

# --- INITIALIZE STATE ---
# Streamlit refreshes the script on every user click. 
# We must save the 'chain' and 'chat history' in Session State so they don't get deleted.
if "strict_mode" not in st.session_state:
    st.session_state.strict_mode = True
if "chain" not in st.session_state:
    st.session_state.chain = setup_rag_chain()
if "messages" not in st.session_state:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            st.session_state.messages = json.load(f)
    else:
        st.session_state.messages = []

# --- SIDEBAR: FILE UPLOAD ---
with st.sidebar:
    st.header("📂 Upload Documents")
    uploaded_file = st.file_uploader("Upload PDF or Excel", type=["pdf", "xlsx"])
    
    if st.button("Process File"):
        if uploaded_file is not None:
            with st.spinner("Processing document..."):
                # Streamlit keeps files in memory. We must save it to a physical temp file for our Data Engine.
                temp_dir = tempfile.mkdtemp()
                temp_filepath = os.path.join(temp_dir, uploaded_file.name)
                
                with open(temp_filepath, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Send the temp file to the Data Engine you built!
                process_document(temp_filepath)
                
                # Re-initialize the brain so it knows about the newly added data
                st.session_state.chain = setup_rag_chain()
                st.session_state.current_file = uploaded_file.name
                st.success(f"✅ {uploaded_file.name} processed and added to the database!")
        else:
            st.warning("Please upload a file first.")

    st.divider()
    st.header("⚙️ Settings")

    # UI: The LLM Provider Selector
    selected_provider = st.sidebar.selectbox(
        "🧠 Select AI Provider",
        ["Groq (Llama 3.1 8B)", "Google (Gemini 2.5 Flash)"]
    )

    # If the user just changed the model. If so, rebuild the chain!
    if "current_provider" not in st.session_state or st.session_state.current_provider != selected_provider:
        st.session_state.current_provider = selected_provider
        with st.spinner(f"Switching engine to {selected_provider}..."):
            # Pass the new choice to the backend
            st.session_state.chain = setup_rag_chain(
                strict_mode=st.session_state.get("strict_mode", True),
                provider=selected_provider
            )

    # UI: The Strict Mode Toggle
    is_strict = st.toggle(
        "🔒 Strict Document Mode", 
        value=st.session_state.strict_mode,
        help="If ON, AI only uses uploaded docs. If OFF, AI can use general knowledge."
    )
    
    # If the user clicks the toggle, we instantly rebuild the AI's brain!
    if is_strict != st.session_state.strict_mode:
        st.session_state.strict_mode = is_strict
        st.session_state.chain = setup_rag_chain(strict_mode=is_strict)
        
        mode_text = "Strict" if is_strict else "Flexible"
        st.toast(f"Switched to {mode_text} Mode!") # Shows a cool little popup notification

    st.divider()
    st.header("⚙️ Data Privacy")
    st.write("Your data is processed locally and never leaves your machine.")

    # UI: The Deletion Button
    if st.sidebar.button("🗑️ Delete My Data"):
        with st.spinner("Erasing database..."):
            
            # 1. BREAK THE RAM LOCK FIRST (Crucial!)
            if "chain" in st.session_state:
                del st.session_state.chain  
                
            if "messages" in st.session_state:
                st.session_state.messages = [] 
                
            # 2. DELETE THE CHROMA FOLDER
            db_path = os.path.join("data", "chroma_db")
            if os.path.exists(db_path):
                shutil.rmtree(db_path)

            # 2. DELETE THE ARCHIVE FOLDER
            archive_path = os.path.join("data", "archive")
            if os.path.exists(archive_path):
                shutil.rmtree(archive_path)

            # 3. DELETE THE JSON HISTORY
            history_path = os.path.join("data", "chat_history.json")
            if os.path.exists(history_path):
                os.remove(history_path)
                
            # 4. REBOOT A BLANK BRAIN
            # Point to your setup function in the src folder
            from src.rag_engine import setup_rag_chain 
            st.session_state.chain = setup_rag_chain(
                strict_mode=st.session_state.get("strict_mode", True)
            )
            
            st.sidebar.success("✅ Database and chat history completely wiped.")

    # 2. Call the cached version instead!
    if "chain" not in st.session_state or st.session_state.current_provider != selected_provider:
        st.session_state.current_provider = selected_provider

        # This will be slow the FIRST time, but instant every time after that.
        st.session_state.chain = get_cached_chain(
            provider=selected_provider
        )

# --- MAIN CHAT INTERFACE ---
# 1. UI: Display past chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. UI: Chat Input Box
if prompt := st.chat_input("Ask a question about your documents..."):
    # Immediately display the user's question on the screen
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Generate the AI Answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and thinking..."):
            # Ask the brain!
            try:
                response = st.session_state.chain.invoke({"question": prompt})
                answer = response["answer"]
                st.write_stream(stream_generator(answer))
        
                # Save the AI's answer to the history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                with open(HISTORY_FILE, "w") as f:
                    json.dump(st.session_state.messages, f)
            except Exception as e:
                error_message = str(e)
                if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
                    st.warning("⏳ **Rate Limit Reached!** The free tier API is cooling down. Please wait about 60 seconds and try again.")
                else:
                    st.error(error_message)
            