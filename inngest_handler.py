import logging

import inngest.fast_api
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client.http.models import UpdateStatus

from pdf_util import load_and_chunk_pdf, embed_chunks_and_save, embed_text, client as gemini_client
from qdrant_helper import search_vectors

inngest_client = inngest.Inngest(
    app_id="rag_testing_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

class RagAndChunkPdf(BaseModel):
    chunks: list[str]
    source_id: str

class RagIdEmbeddingsSaveStatus(BaseModel):
    save_status: UpdateStatus

@inngest_client.create_function(
    fn_id="rag_ingest",
    retries = 2,
    trigger=inngest.TriggerEvent(event="rag/pdf_uploaded"),
)
async def rag_ingest(ctx: inngest.Context):

    def load_and_convert_to_chunks(ctx: inngest.Context) -> RagAndChunkPdf:
        file_path = ctx.event.data["file_path"]
        chunks = load_and_chunk_pdf(file_path)
        return RagAndChunkPdf(chunks = chunks, source_id = file_path)

    def chunks_to_embeddings(rag_and_chunks: RagAndChunkPdf) -> RagIdEmbeddingsSaveStatus:
        save_status = embed_chunks_and_save(rag_and_chunks.chunks, rag_and_chunks.source_id)
        return RagIdEmbeddingsSaveStatus(save_status = save_status)

    rag_and_chunks =  await ctx.step.run("load_and_convert_to_chunks", lambda: load_and_convert_to_chunks(ctx), output_type = RagAndChunkPdf)
    ingested =  await ctx.step.run("chunks_to_embeddings", lambda: chunks_to_embeddings(rag_and_chunks), output_type = RagIdEmbeddingsSaveStatus)

    return ingested.model_dump()

@inngest_client.create_function(
    fn_id="rag_query",
    retries = 2,
    trigger=inngest.TriggerEvent(event="rag/query_data"),
)
async def rag_query(ctx: inngest.Context):

    def embed_and_search(ctx: inngest.Context) -> list[dict[str, str]]:
        qstn = ctx.event.data["question"]
        query_vector = next(embed_text([qstn])).tolist()
        res = search_vectors(query_vector)
        score_threshold = 0.45

        matching_chunks = [
            {"content": point.payload["text"],
             "source_id": point.payload["source_id"], }
            for point in res.points
            if point.payload
               and "text" in point.payload
               and getattr(point, "score", 0) >= score_threshold
        ]

        return matching_chunks

    def ask_llm(context_chunks: list[str], question: str) -> str:

        prompt = (
            f"Answer the following question based only on the provided context.\n\n"
            f"Context:\n{context_chunks}\n\n"
            f"Question: {question}\n\n"
            f"Explain [Topic] in under 40 words.\n\n"
            f"Summarize this in 4-5 bullet points.\n\n"
            f"Answer:"
        )
        llm_response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return llm_response.text

    question = ctx.event.data["question"]

    context_chunks = await ctx.step.run("embed_and_search_question", lambda: embed_and_search(ctx), output_type=list[dict[str, str]])
    answer = await ctx.step.run("ask_llm", lambda: ask_llm(context_chunks, question), output_type=str)
    sources = [chunk['source_id'] for chunk in context_chunks]

    return {"answer": answer, "sources": sources}


app = FastAPI()

# Serve the Inngest endpoint
inngest.fast_api.serve(app, inngest_client, [rag_ingest, rag_query])