from langchain_openai import ChatOpenAI
import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_xai import ChatXAI
from langchain_anthropic import ChatAnthropic

load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")
together_api = os.getenv("TOGETHER_API_KEY")
openai_api = os.getenv("OPENAI_API_KEY")
xai_api = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
anthropic_api = os.getenv("ANTHROPIC_API_KEY")

# Provider selection: "openai", "anthropic", or "xai"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

LLM_EXTRACTION_MODEL = os.getenv("LLM_EXTRACTION_MODEL", "gpt-5.2-2025-12-11")
# PDF rendering quality (DPI) - affects all extractions
LLM_PDF_RENDER_DPI = int(os.getenv("LLM_PDF_RENDER_DPI", "220"))

# Set environment variables for provider code to read
os.environ["LLM_PROVIDER"] = LLM_PROVIDER
os.environ["LLM_EXTRACTION_MODEL"] = LLM_EXTRACTION_MODEL
os.environ["LLM_PDF_RENDER_DPI"] = str(LLM_PDF_RENDER_DPI)

client = OpenAI(api_key=openai_api)

# Grading provider/model can be changed from env without code edits.
# Supported providers: openai, anthropic, xai
GRADING_PROVIDER = os.getenv("GRADING_PROVIDER", "openai").strip().lower()
LLM_GRADER_MODEL = os.getenv("LLM_GRADER_MODEL", "gpt-5.2-2025-12-11")
LLM_CHAIN_MODEL = os.getenv("LLM_CHAIN_MODEL", LLM_GRADER_MODEL)


def _build_chat_model(provider: str, model: str, temperature: float = 0):
    if provider == "openai":
        return ChatOpenAI(
            model=model,
            api_key=openai_api,
            temperature=temperature,
        )
    if provider in ["anthropic", "claude"]:
        return ChatAnthropic(
            model=model,
            api_key=anthropic_api,
            temperature=temperature,
            max_retries=2,
        )
    if provider in ["xai", "grok"]:
        return ChatXAI(
            model=model,
            xai_api_key=xai_api,
            temperature=temperature,
            max_retries=2,
        )
    raise ValueError(
        f"Unsupported GRADING_PROVIDER '{provider}'. "
        "Supported: openai, anthropic, xai"
    )


llm_grader = _build_chat_model(GRADING_PROVIDER, LLM_GRADER_MODEL, temperature=0)
llm = _build_chat_model(GRADING_PROVIDER, LLM_CHAIN_MODEL, temperature=0)
