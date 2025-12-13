from langchain_openai import ChatOpenAI
import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_xai import ChatXAI

load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")
together_api = os.getenv("TOGETHER_API_KEY")
openai_api = os.getenv("OPENAI_API_KEY")
grok_api = os.getenv("GROK_API_KEY")


client = OpenAI(api_key=openai_api)

# llm_grader = ChatOpenAI(
#     # model="gpt-5-2025-08-07",
#     model = "gpt-5.1-2025-11-13",
#     api_key=openai_api,
#     temperature=0
# )

llm_grader = ChatXAI(
    model="grok-4-1-fast-reasoning",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=grok_api
)

llm = ChatOpenAI(
    model="gpt-5-mini-2025-08-07",
    api_key=openai_api,
    temperature=0
)
