def call_llm(client, model: str, prompt: str, max_tokens: int = 1024) -> str:
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
