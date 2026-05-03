"""Builds the LangChain RAG pipeline: history-aware retriever + answer generator."""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from app.config import settings


SYSTEM_PROMPT = """You are Orbis Iris Accelerator's documentation assistant. Answer the user's
question using ONLY the provided context excerpts. If the context does not
contain the answer, say you don't know — do not invent details.

Cite your sources inline using bracketed numbers like [1], [2] that map to the
context excerpts in the order given. Keep answers concise and structured;
prefer bullet lists for steps and short paragraphs for explanations.

Context:
{context}"""

REPHRASE_PROMPT = """Given the chat history and the latest user question, rewrite the question
to be a standalone search query. If the latest question is already standalone, return it as is.
Output only the rewritten query."""

REFUSAL = (
    "I can only answer questions about the Iris Accelerator documentation. "
    "Try asking about its architecture, services, infrastructure, device manager, "
    "IMS SDK, or troubleshooting."
)


# --- small-talk routing -----------------------------------------------------
# Pure greetings / farewells / thanks should not be sent through retrieval at all.
# They would otherwise score below SIMILARITY_THRESHOLD and get the refusal text,
# which feels rude. We match conservatively: only short messages that are
# *entirely* small talk; mixed messages like "hi, what services run on iris?"
# still go through normal retrieval.

_GREETINGS = {
    "hi", "hello", "hey", "yo", "hola", "howdy", "greetings",
    "hi there", "hello there", "hey there",
    "good morning", "good afternoon", "good evening", "good day",
    "morning", "afternoon", "evening",
}
_FAREWELLS = {
    "bye", "goodbye", "good bye", "see you", "see ya", "see you later",
    "take care", "later", "cya", "ttyl", "have a good one", "have a nice day",
}
_THANKS = {
    "thanks", "thank you", "thank you very much", "thanks a lot",
    "ty", "thx", "appreciate it", "much appreciated", "cheers",
}


def detect_smalltalk(text: str) -> str | None:
    """Return 'greeting' | 'farewell' | 'thanks' if the message is purely
    small talk; otherwise None."""
    t = text.strip().lower().rstrip(".!?,")
    if not t:
        return None
    if len(t.split()) > 6:
        return None
    if t in _GREETINGS:
        return "greeting"
    if t in _FAREWELLS:
        return "farewell"
    if t in _THANKS:
        return "thanks"
    return None


def smalltalk_reply(kind: str) -> str:
    if kind == "greeting":
        return (
            "Hi! I'm here to answer questions about the **Iris Accelerator** "
            "documentation — architecture, services, IMS SDK, troubleshooting, "
            "and so on. What would you like to know?"
        )
    if kind == "farewell":
        return "Goodbye! Come back anytime you have questions about the Iris Accelerator."
    if kind == "thanks":
        return "You're welcome! Anything else you'd like to know about the Iris Accelerator?"
    return ""


def build_rag_chain(llm, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.TOP_K})

    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", REPHRASE_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, rephrase_prompt)

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, answer_prompt)

    return create_retrieval_chain(history_aware_retriever, qa_chain)


def is_in_scope(vectorstore, query: str) -> tuple[bool, float]:
    """Return (in_scope, top_cosine_score) for the raw user query.

    Vectorstore was built with normalize_L2=True + MAX_INNER_PRODUCT,
    so similarity_search_with_score returns cosine similarity.
    """
    results = vectorstore.similarity_search_with_score(query, k=1)
    if not results:
        return False, 0.0
    top_score = max(0.0, float(results[0][1]))
    return top_score >= settings.SIMILARITY_THRESHOLD, top_score
