"""LangChain LLM factory. Adding a new provider = adding one branch."""
from langchain_core.language_models.chat_models import BaseChatModel
from app.config import settings


def build_llm() -> BaseChatModel:
    provider = settings.LLM_PROVIDER.lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,
            streaming=True,
            temperature=0.2,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,
            streaming=True,
            temperature=0.2,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,
            streaming=True,
            temperature=0.2,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=settings.LLM_MODEL, temperature=0.2)

    raise ValueError(
        f"Unknown LLM_PROVIDER: {settings.LLM_PROVIDER!r}. "
        f"Supported: groq, openai, anthropic, ollama."
    )
