from groq import Groq


def generate_answer(query, reranked_chunks, api_key):
    """
    Generate the final answer using Groq, based only on the provided
    reranked chunks (context). Instructs the model not to use outside
    knowledge and to say so clearly if the answer isn't in the context.
    """
    client = Groq(api_key=api_key)

    context = "\n\n".join([chunk["text"] for chunk in reranked_chunks])

    system_prompt = (
        "You are a legal assistant answering questions based only on the "
        "provided contract excerpts. Do not use outside knowledge. "
        "If the answer is not found in the provided context, say so clearly."
    )

    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )

    return response.choices[0].message.content
