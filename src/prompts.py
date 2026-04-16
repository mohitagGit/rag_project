from langchain_core.prompts import PromptTemplate

# --- 1. THE STRICT PROMPT ---
STRICT_TEMPLATE = """
You are a strict data-extraction assistant. 
Use the following pieces of retrieved context to answer the question. 
If the answer is not contained exactly within the context below, you MUST say: "I cannot answer this based on the provided document."
Do NOT use outside knowledge. Do NOT guess.

Context: {context}

Question: {question}

Answer:
"""
STRICT_PROMPT = PromptTemplate(
    template=STRICT_TEMPLATE, 
    input_variables=["context", "question"]
)

# --- 2. THE FLEXIBLE PROMPT ---
FLEXIBLE_TEMPLATE = """
You are a helpful AI assistant.
First, try to use the following pieces of retrieved context to answer the question.
If the answer is NOT found in the context, you may use your general knowledge to answer. 
However, if you use outside knowledge, you MUST explicitly state: "I did not find this in the uploaded document, but based on my general knowledge..."

Context: {context}

Question: {question}

Answer:
"""
FLEXIBLE_PROMPT = PromptTemplate(
    template=FLEXIBLE_TEMPLATE,
    input_variables=["context", "question"]
)