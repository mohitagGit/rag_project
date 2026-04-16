import os
import shutil
from datetime import datetime
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# We define the folder where our Chroma DB will be saved
os.makedirs("data", exist_ok=True)
CHROMA_PATH = os.path.join("data", "chroma_db")

def process_document(file_path):
    """
    Takes a file path (PDF or XLSX), extracts text, chunks it, and saves to Vector DB.
    """
    print(f"📄 Processing {file_path}...")
    documents = []
    
    # --- STEP 1: PARSING ---
    if file_path.endswith('.pdf'):
        # Load PDF text
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
    elif file_path.endswith('.xlsx'):
        # Load Excel and convert rows to text strings (Handling the "Table Problem")
        df = pd.read_excel(file_path)
        text_data = ""
        for index, row in df.iterrows():
            row_text = ", ".join([f"{col}: {val}" for col, val in row.items()])
            text_data += row_text + "\n"
        
        # We wrap the string in a Langchain Document object so the chunker can read it
        from langchain.schema.document import Document
        documents = [Document(page_content=text_data, metadata={"source": file_path})]
    
    else:
        raise ValueError("Unsupported file type. Please upload PDF or XLSX.")

    # --- DATA CLEANING ---
    # Strip out the artificial PDF line breaks so the chunker works mathematically
    for doc in documents:
        doc.page_content = doc.page_content.replace('\n', ' ')

    # --- STEP 2: CHUNKING ---
    # We use overlapping chunks to prevent "Sentence Cut-Off" context loss
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,     # Size of each chunk in characters
        chunk_overlap=100,  # Characters to overlap with the next chunk
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ Split file into {len(chunks)} chunks.")

    # --- STEP 3: EMBEDDING & VECTOR DB ---
    # We use a free, local HuggingFace model (doesn't send data to OpenAI)
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # FIX: Sanitize metadata before giving it to Chroma
    for chunk in chunks:
        clean_metadata = {}
        for key, value in chunk.metadata.items():
            # Chroma only accepts strings, ints, and floats.
            if isinstance(value, (str, int, float)):
                clean_metadata[key] = value
            elif value is None:
                continue # Skip null values
            else:
                # Force anything weird into a plain string
                clean_metadata[key] = str(value) 
        
        chunk.metadata = clean_metadata

    # Save the chunks and their mathematical vectors to the local database
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model, 
        persist_directory=CHROMA_PATH
    )
    db.persist()
    print("✅ Successfully embedded and saved to Chroma DB!")

    # --- STEP 4: ARCHIVE THE FILE ---
    # 1. Ensure the archive folder exists in your project directory
    os.makedirs("data", exist_ok=True)
    archive_folder = os.path.join("data", "archive")
    if not os.path.exists(archive_folder):
        os.makedirs(archive_folder)
        print("📁 Created 'archive' folder.")

    # 2. Move the processed file to the archive
    original_filename = os.path.basename(file_path)
    shutil.move(file_path, os.path.join(archive_folder, original_filename))
    print(f"📂 Moved file {original_filename} to archive.")

def delete_user_data(filename):
    """
    Permanently deletes a specific file and its vector data.
    """
    print(f"🗑️ Initiating deletion protocol for: {filename}...")

    # --- 1. DELETE FROM CHROMA DB ---
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
    
    # We query the database to find the exact IDs of chunks that came from this file
    # (This uses the metadata we attached during Step 1 of parsing!)
    matching_chunks = db.get(where={"source": filename})
    
    if matching_chunks['ids']:
        # Delete those specific chunks by their IDs
        db.delete(ids=matching_chunks['ids'])
        print(f"   -> Deleted {len(matching_chunks['ids'])} chunks from Vector DB.")
    else:
        print("   -> No chunks found in Vector DB.")

    # --- 2. DELETE FROM ARCHIVE FOLDER ---
    archive_path = os.path.join("data", "archive", filename)
    if os.path.exists(archive_path):
        os.remove(archive_path)
        print("   -> Deleted physical file from archive.")
    else:
        print("   -> Physical file not found in archive.")

    print("✅ User data completely scrubbed from the system.")