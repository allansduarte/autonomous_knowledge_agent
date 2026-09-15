from langchain_core.messages import SystemMessage
from agentic.tools import get_ticket_details, update_ticket_status, log_ticket_message

def get_ticket_agent_prompt():
    return SystemMessage(
        content=(
            "You are the Udahub Ticket Management Agent. Your role is to update ticket metadata, set issue tags, "
            "change ticket status (e.g. 'open', 'pending', 'escalated', 'resolved'), and log conversation entries. "
            "Use `get_ticket_details`, `update_ticket_status`, and `log_ticket_message` tools to manage ticket lifecycles."
        )
    )

def create_ticket_tools():
    return [get_ticket_details, update_ticket_status, log_ticket_message]
