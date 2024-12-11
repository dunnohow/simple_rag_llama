import os
import time
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama
import sqlite3
from typing import List, Tuple
from PyPDF2 import PdfReader
import logging
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

# Initialize Flask app
app = Flask(__name__)
logger = logging.getLogger("sample llm task")

# Initialize the embedding model and LLM
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
llm = Llama(model_path="/path/to/llama/file",
            n_ctx=5000
            )  # Replace with actual path to model

# Database setup
db_file = "chatbot_responses.db"
if not os.path.exists(db_file):
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            response TEXT NOT NULL,
            latency REAL NOT NULL
        )
        """)

# Load paper content and create embeddings
def extract_text_from_pdf(pdf_path):
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text()
    return text

def segment_embed_text(text, chunk_size=500):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    embeddings = [(chunk, embedding_model.encode(chunk, convert_to_tensor=True)) for chunk in chunks]
    return embeddings

imported_text = extract_text_from_pdf("./llama2 paper.pdf")
paper_embeddings = segment_embed_text(imported_text)

# Retrieve relevant context
def retrieve_context(query: str, top_k: int = 3) -> List[str]:
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    similarities = [(chunk, util.pytorch_cos_sim(query_embedding, embedding)[0][0].item()) for chunk, embedding in paper_embeddings]
    top_chunks = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    return [chunk for chunk, _ in top_chunks]

# Query chatbot
@app.route("/query", methods=["POST"])
def query():
    data = request.json
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "Question not provided."}), 400

    start_time = time.time()

    relevant_context = "\n".join(retrieve_context(question))
    llm_input = (f"You are an advanced answering assistant trained on an extensive corpus of text. "
                f"Your goal is to respond accurately and professionally based on the given context. "
                f"Ensure all answers are clear, concise, and relevant to the provided context. "
                f"Do not attempt to answer questions outside the scope of the context. "
                f"Use professional, straightforward language to convey your responses. "
                f"Context: {relevant_context}\n"
                f"Question: {question}"
                )

    # logger.info("test logging")
    llm_response = llm(
        llm_input,
        max_tokens=100,  # Maximum number of tokens to generate
        temperature=0.7,  # Creativity level
    )['choices'][0]['text']
    # logger.info(llm_response)

    latency = time.time() - start_time

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO responses (question, response, latency) VALUES (?, ?, ?)", (question, llm_response, latency))
        conn.commit()

    return jsonify({"response": llm_response, "latency": latency})

# Retrieve stored responses
@app.route("/responses", methods=["GET"])
def get_responses():
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM responses")
        rows = cursor.fetchall()
        responses = [{"id": row[0], "question": row[1], "response": row[2], "latency": row[3]} for row in rows]
    return jsonify(responses)

if __name__ == "__main__":
    app.run(debug=True)
