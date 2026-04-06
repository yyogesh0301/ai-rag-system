from rag.retrieve import retrieve
from rag.providers import get_provider

provider = get_provider()

chat_history = []


def ask(question: str) -> str:
    context_chunks = retrieve(question, k=5)
    context = "\n".join(context_chunks)

    history_text = ""
    for q, a in chat_history:
        history_text += f"User: {q}\nAI: {a}\n"

    prompt = f"""You are a helpful assistant.
    Use the context if relevant, otherwise answer from your own knowledge

Previous conversation:
{history_text}

Context:
{context}

Question:
{question}"""

    answer = provider.generate(prompt)
    chat_history.append((question, answer))
    return answer
