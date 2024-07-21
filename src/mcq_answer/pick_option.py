import re
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from indexing.enrich import DocumentProcessor
from retrieval.storage import VectorRetrieval


def initialize_llm():
    local_model = "phi3"
    return ChatOllama(model=local_model)


def create_prompt_template():
    template = """Pick the right option answering the question based ONLY on the following context:
{context}
#########
{question}
#########
Give your answer in the following format:
option<number>: <your only one answer option>
explanation: <small comprehensive explanation>
"""
    return ChatPromptTemplate.from_template(template)


def load_vector_retrieval(embedding_model, persist_directory):
    return VectorRetrieval(embedding_model, persist_directory)


def load_document_processor(file_path):
    return DocumentProcessor(file_path)


def create_chain(llm, prompt, retriever):
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def answer_question(query, option1, option2, option3, option4):
    # Initialize components
    llm = initialize_llm()
    prompt = create_prompt_template()
    vr = load_vector_retrieval("all-MiniLM-L6-v2", "chroma")
    dp = load_document_processor(r"data/3GPP_vocabulary.docx")

    # Process the query
    enriched_query = dp.define_TA_question(query)
    db = vr._load_from_disk()
    retriever = db.as_retriever(search_type="mmr")
    chain = create_chain(llm, prompt, retriever)

    # Format the options
    options_text = f"1){option1} 2){option2} 3){option3} 4){option4}"

    # Invoke the chain
    result = chain.invoke(f"the question is : {enriched_query} the options are : {options_text}")
    return result


def get_option_and_explanation(result):
    # Define regex patterns
    option_pattern = r'\boption\d+\s*:\s*[^:\n]+'
    explanation_pattern = r'\bexplanation\s*:\s*.+'

    # Find matches for options
    answer = result.lower()
    option = re.findall(option_pattern, answer, re.IGNORECASE)

    # Find the explanation
    explanation_match = re.search(explanation_pattern, answer, re.IGNORECASE)
    if explanation_match:
        explanation = explanation_match.group()

    return option, explanation


if __name__ == "__main__":
    query = "When can the setting of the Privacy exception list be changed?"
    option1 = "Never"
    option2 = "Only during emergency services"
    option3 = "Anytime"
    option4 = "Only with operator permission"

    answer = answer_question(query, option1, option2, option3, option4)
    print("------------------")
    print(answer)
