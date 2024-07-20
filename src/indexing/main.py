import indexing
from langchain_core.documents.base import Document


def main():
    folder_path = 'data/knowledge_base/'
    file_path = r"data/3GPP_vocabulary.docx"
    persist_directory = "chroma"
    embedding_model = "all-MiniLM-L6-v2"

    doc_loader = indexing.loading.DocumentLoader(folder_path)
    documents = doc_loader.load_documents()

    text_splitter = indexing.chunking.TextSplitter(chunk_size=2000, chunk_overlap=200, word_split=True)
    document_chunker = indexing.chunking.DocumentChunker(text_splitter=text_splitter)

    doc_processor = indexing.enrich.DocumentProcessor(file_path)

    processor = indexing.preprocessing.TextProcessor()

    vsi = indexing.storage.VectorStoreIngestor(embedding_model, persist_directory)

    docs = []
    for doc in documents:
        chunks = document_chunker.chunk_doc(doc)
        for chunk in chunks:
            enriched_chunk = doc_processor.define_TA_question(chunk.page_content)
            docs.append(Document(page_content=enriched_chunk,
                                 metadata=chunk.metadata))
    print(f"documents are ready to preprocess with number : {len(docs)}")
    processed_docs = processor.preprocess_documents(docs)
    print(f"processed document example : {processed_docs[6]}")
    vsi.ingest(processed_docs)


if __name__ == "__main__":
    main()
