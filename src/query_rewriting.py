from groq import Groq

def rewrite_query(query, api_key):
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistant specialized in rephrasing legal questions "
                    "to make them clearer and more precise for searching a legal "
                    "contracts database. Rephrase the user's question using "
                    "alternative legal terminology where relevant. If the question "
                    "is in any language other than English, rephrase it in English. "
                    "Return only the rephrased question, with no explanation or "
                    "additional text."
                )
            },
            {
                "role": "user",
                "content": query
            }
        ]
    )

    return response.choices[0].message.content