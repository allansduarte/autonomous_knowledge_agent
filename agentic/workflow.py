from typing import Annotated, Sequence, TypedDict
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from agentic.agents import (
    get_supervisor_agent_prompt,
    get_support_agent_prompt,
    get_account_agent_prompt,
    get_ticket_agent_prompt,
)
from agentic.tools import (
    search_knowledge_base,
    get_user_profile,
    get_user_reservations,
    search_experiences,
    get_ticket_details,
    update_ticket_status,
    log_ticket_message,
    get_customer_history,
    escalate_ticket,
)

load_dotenv()

# 1. Define State Schema
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# 2. Register Tools
tools = [
    search_knowledge_base,
    get_user_profile,
    get_user_reservations,
    search_experiences,
    get_ticket_details,
    update_ticket_status,
    log_ticket_message,
    get_customer_history,
    escalate_ticket,
]

# 3. Model with Tools bound
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY") or "sk-dummy-key-for-init"
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
model_with_tools = model.bind_tools(tools)

# 4. Core System Prompt combining Agent Instructions
system_prompt = SystemMessage(
    content=(
        "You are the Uda-hub CultPass Multi-Agent Support Orchestrator.\n"
        "Your goal is to provide fast, helpful, accurate, grounded support for CultPass members.\n\n"
        "Agent Roles & Workflow Instructions:\n"
        "1. **Knowledge & FAQ Support**: Use `search_knowledge_base` to retrieve authoritative articles. "
        "   - If `should_escalate` is True or confidence is low, use `escalate_ticket` to hand over to a human lead.\n"
        "2. **Account & Reservation Services**: Use `get_user_profile`, `get_user_reservations`, and `search_experiences` to inspect user data.\n"
        "3. **Ticket & Long-Term Memory**: Use `get_customer_history` to inspect past resolved tickets across sessions. "
        "   - Use `update_ticket_status` and `log_ticket_message` to maintain ticket status and persist history.\n"
        "Always personalize answers based on customer details and remain polite and grounded."
    )
)

# 5. Agent Node Function
def agent_node(state: AgentState):
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [system_prompt] + list(messages)
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

# 6. Build LangGraph StateGraph from scratch
builder = StateGraph(AgentState)

# Add Nodes
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

# Add Edges
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition, ["tools", END])
builder.add_edge("tools", "agent")

# Compile Graph with MemorySaver checkpointer for thread_id session persistence
checkpointer = MemorySaver()
orchestrator = builder.compile(checkpointer=checkpointer)