from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma


class VectorRetrieval:
    def __init__(self, embedding_model: str, persist_directory: str):
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model

    def _load_from_disk(self) -> Chroma:
        embedding_function = SentenceTransformerEmbeddings(
            model_name=self.embedding_model
        )
        db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embedding_function
        )
        return db

    def retrieve_documents(self, enriched_question):
        db = self._load_from_disk()
        retriever = db.as_retriever(search_type="mmr")
        return retriever.invoke(enriched_question)
