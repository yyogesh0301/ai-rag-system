from google import genai
import os
from dotenv import load_dotenv
from rag.retrieve import retrieve

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

chat_history = []

def ask(question: str) -> str:

    context_chunks = retrieve(question, k=5)
    context = "\n".join(context_chunks)

    #print("\n--- Retrieved Context ---")
    # for c in context_chunks:
    #     print(c[:900])
    # print(f"{context}")

    history_text = ""
    for q, a in chat_history:
        history_text += f"User: {q}\nAI: {a}\n"

    prompt = f"""
        You are a helpful assistant.

        ONLY answer using the provided context.
        If the answer is not in the context, say "I don't know".

        If the question asks for:
        - name → return exact name
        - date → return exact date
        - number → return exact value

        DO NOT summarize unnecessarily.

        Context:
        {context}

        Question:
        {question}
        """

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    answer = response.text

    chat_history.append((question, answer))

    return answer