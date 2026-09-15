from langchain_core.messages import SystemMessage
from agentic.tools import get_user_profile, get_user_reservations, search_experiences

def get_account_agent_prompt():
    return SystemMessage(
        content=(
            "You are the CultPass Account & Reservation Agent. Your role is to inspect user account details, "
            "subscription status, quotas, and existing reservations, and assist with experience searches. "
            "Use `get_user_profile`, `get_user_reservations`, and `search_experiences` tools to retrieve accurate account state."
        )
    )

def create_account_tools():
    return [get_user_profile, get_user_reservations, search_experiences]
