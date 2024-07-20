from typing import List
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents.base import Document


class VectorStoreIngestor:
    def __init__(self, embedding_model: str, persist_directory: str):
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model

    def batch_data(self, data: List[Document], batch_size: int):
        """Splits data into batches of the specified size."""
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    def ingest(self, documents: List[Document], batch_size: int = 5400):
        embedding_function = SentenceTransformerEmbeddings(
            model_name=self.embedding_model
        )

        for batch in self.batch_data(documents, batch_size):
            Chroma.from_documents(
                documents=batch,
                embedding=embedding_function,
                persist_directory=self.persist_directory
            )
            print(f"Ingested {len(batch)} documents into {self.persist_directory}")
