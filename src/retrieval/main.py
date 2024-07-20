from indexing.enrich import DocumentProcessor
from retrieval.storage import VectorRetrieval


def main(query):
    persist_directory = "chroma"
    embedding_model = "all-MiniLM-L6-v2"
    file_path = r"data/3GPP_vocabulary.docx"

    vr = VectorRetrieval(embedding_model, persist_directory)
    dp = DocumentProcessor(file_path)
    enriched_query = dp.define_TA_question(query)
    context = vr.retrieve_documents(enriched_query)
    print(context)


if __name__ == "__main__":
    query = "What is the purpose of the Nmfaf_3daDataManagement_Deconfigure service operation?"
    main(query)
