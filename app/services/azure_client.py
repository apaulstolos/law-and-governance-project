from __future__ import annotations

from typing import List

from openai import AzureOpenAI

from app.config import get_settings


settings = get_settings()

_client = AzureOpenAI(
    azure_endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_api_key,
    api_version=settings.azure_openai_api_version,
)


def embed_texts(texts: List[str]) -> List[List[float]]:
    response = _client.embeddings.create(
        input=texts,
        model=settings.azure_openai_embedding_deployment,
    )
    return [item.embedding for item in response.data]


def generate_summary(prompt: str, max_tokens: int = 200) -> str:
    completion = _client.chat.completions.create(
        model=settings.azure_openai_chat_deployment,
        messages=[
            {
                "role": "system",
                "content": "You summarize public comments related to environmental impact statements.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content.strip()


def generate_response(prompt: str, max_tokens: int = 350) -> str:
    completion = _client.chat.completions.create(
        model=settings.azure_openai_chat_deployment,
        messages=[
            {
                "role": "system",
                "content": "You draft concise, respectful agency responses to public comments.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content.strip()
