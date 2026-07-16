from langchain_google_genai import ChatGoogleGenerativeAI
from app.config.config import LLM_MODEL, TEMPERATURE, MAX_OUTPUT_TOKENS

def get_llm() -> ChatGoogleGenerativeAI:
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS
    )
    return llm