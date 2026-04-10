from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def build_chain(retriever, llm):
    """Build an LCEL RAG chain: retrieve → prompt → llm → parse."""

    prompt = ChatPromptTemplate.from_template("""You are a helpful assistant.
Use the context below if it is relevant to the question.
If not, answer from your own knowledge.

Context:
{context}

Question:
{question}""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
