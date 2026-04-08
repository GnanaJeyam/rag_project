from uuid import uuid5, NAMESPACE_URL

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, UpdateStatus, PointStruct, QueryResponse

collection_name = "rag_vectors"
vector_size = 384
qdrant_client = QdrantClient("localhost", port=6333)


def create_collection_if_not_exists():
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    return True

def insert_vector(embeds: list[list[float]], chunks: list[str], source_id: str) -> UpdateStatus:
    create_collection_if_not_exists()
    points = [
        PointStruct(
            id=str(uuid5(NAMESPACE_URL, f"{source_id}:{i}:{chunk}")),
            vector=emb,
            payload={
                "text": chunk,
                "source_id": source_id,
                "chunk_index": i,
            },
        )
        for i, (emb, chunk) in enumerate(zip(embeds, chunks))
    ]

    qdrant_status = qdrant_client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True
    )

    return qdrant_status.status

def search_vectors(query_vector) -> QueryResponse:
    create_collection_if_not_exists()
    return qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=5,
        with_payload=True,
    )
