import logging
from pathlib import Path

import google.genai as geni
from dotenv import load_dotenv
from fastembed import TextEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader

import qdrant_helper

load_dotenv()
sentence_splitter = SentenceSplitter(chunk_size=800, chunk_overlap=200)
text_embedding = TextEmbedding()
client = geni.Client()
vector_size = 384


def save_pdf(uploaded_file):
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    uploaded_file_name = uploads_dir / uploaded_file.name
    uploaded_file_name.write_bytes(uploaded_file.getbuffer())
    return uploaded_file_name


def load_and_chunk_pdf(file_path):
    text_data = PDFReader().load_data(file_path)
    chunks = []
    for node in text_data:
        current_word = getattr(node, "text", None)
        if current_word:
            chunks.extend(sentence_splitter.split_text(current_word))

    logging.info("Chunks created for this file: %s", file_path)

    return chunks

def embed_text(chunks):
    return text_embedding.embed(
        chunks,
        batch_size=vector_size,
        parallel=None,
        dimension=vector_size,
    )

def embed_chunks_and_save(chunks: list[str], source_id: str):
    embed_response = embed_text(chunks)
    result = [r.tolist() for r in embed_response]

    qdrant_status = qdrant_helper.insert_vector(result, chunks, source_id)

    logging.info("Qdrant status %s", qdrant_status)

    return qdrant_status
