from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from agentic.tools import search_knowledge_base

def get_support_agent_prompt():
    return SystemMessage(
        content=(
            "You are the CultPass Knowledge Support Agent. Your role is to answer user questions regarding policies, "
            "subscriptions, venue access, rules, cancellations, and troubleshooting. "
            "Always use the `search_knowledge_base` tool to look up authoritative articles before answering. "
            "Be clear, polite, and directly address the user's inquiry."
        )
    )

def create_support_tools():
    return [search_knowledge_base]
