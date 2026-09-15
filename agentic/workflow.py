from typing import Annotated, Sequence, TypedDict, Optional, Dict, Any
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from agentic.agents import (
    get_supervisor_agent_prompt,
    get_support_agent_prompt,
    get_account_agent_prompt,
    get_ticket_agent_prompt,
    create_support_tools,
    create_account_tools,
    create_ticket_tools,
)
from agentic.logger import log_event

load_dotenv()

# 1. State Schema
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    ticket_metadata: Optional[Dict[str, Any]]

# 2. Specialist Tools Registration
support_tools = create_support_tools()
account_tools = create_account_tools()
ticket_tools = create_ticket_tools()
all_tools = support_tools + account_tools + ticket_tools

# 3. Models bound with respective tools
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY") or "sk-dummy-key-for-init"
base_model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

supervisor_model = base_model.bind_tools(all_tools)
support_model = base_model.bind_tools(support_tools)
account_model = base_model.bind_tools(account_tools)
ticket_model = base_model.bind_tools(ticket_tools)

# 4. Ticket Classification Function
def classify_ticket(message_text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Classifies incoming tickets based on content and metadata (urgency, tags, status, complexity).
    Returns a dictionary containing selected_agent, category, urgency, and routing_reason.
    """
    metadata = metadata or {}
    text_lower = (message_text or "").lower()
    tags = str(metadata.get("tags", "")).lower()
    urgency = str(metadata.get("urgency", "normal")).lower()
    status = str(metadata.get("status", "")).lower()
    
    # 1. Complex / High Urgency / Ticket Management Issues -> Ticket Agent
    ticket_keywords = ["ticket", "status", "history", "log", "escalat", "issue_type", "dispute", "urgent", "human", "blocked account"]
    if urgency in ["high", "urgent", "critical"] or any(k in tags for k in ["billing", "escalated", "dispute", "fraud"]) or any(k in text_lower for k in ticket_keywords):
        return {
            "selected_agent": "ticket_agent",
            "category": "complex",
            "urgency": "high" if urgency in ["high", "urgent", "critical"] or "dispute" in text_lower else "medium",
            "routing_reason": "Routed to Ticket Agent due to high urgency, metadata tags, escalation request, or ticket history lookup requirement."
        }
    
    # 2. Account / Profile / Quotas / Reservations -> Account Agent
    account_keywords = ["profile", "reservation", "quota", "experience", "tier", "user_id", "subscription status", "my pass", "book", "event catalog"]
    if any(k in tags for k in ["profile", "reservation", "account", "quota"]) or any(k in text_lower for k in account_keywords):
        return {
            "selected_agent": "account_agent",
            "category": "account",
            "urgency": "normal",
            "routing_reason": "Routed to Account Agent due to account profile, reservation, tier, or experience search request."
        }
        
    # 3. Default: Policy / FAQ / How-To / General Questions -> Support Agent
    return {
        "selected_agent": "support_agent",
        "category": "policy",
        "urgency": "normal",
        "routing_reason": "Routed to Knowledge Support Agent for grounded policy FAQ and general knowledge lookup."
    }

# 5. Agent Executable Node Functions
def supervisor_agent_node(state: AgentState):
    """Supervisor Agent: inspects request, formats system prompt, and executes supervisor model."""
    messages = state["messages"]
    ticket_id = state.get("ticket_metadata", {}).get("ticket_id", "default_thread")
    log_event(ticket_id=ticket_id, event_type="AGENT_EXECUTION", agent_name="supervisor_agent")
    
    sup_prompt = get_supervisor_agent_prompt()
    clean_messages = [sup_prompt] + [m for m in messages if not isinstance(m, SystemMessage)]
    response = supervisor_model.invoke(clean_messages)
    return {"messages": [response]}

def support_agent_node(state: AgentState):
    """Support Specialist Agent: executes KB search and policy inquiries using get_support_agent_prompt()."""
    messages = state["messages"]
    ticket_id = state.get("ticket_metadata", {}).get("ticket_id", "default_thread")
    log_event(ticket_id=ticket_id, event_type="AGENT_EXECUTION", agent_name="support_agent")
    
    supp_prompt = get_support_agent_prompt()
    clean_messages = [supp_prompt] + [m for m in messages if not isinstance(m, SystemMessage)]
    response = support_model.invoke(clean_messages)
    return {"messages": [response]}

def account_agent_node(state: AgentState):
    """Account Specialist Agent: executes profile, subscription, reservation, and experience searches using get_account_agent_prompt()."""
    messages = state["messages"]
    ticket_id = state.get("ticket_metadata", {}).get("ticket_id", "default_thread")
    log_event(ticket_id=ticket_id, event_type="AGENT_EXECUTION", agent_name="account_agent")
    
    acc_prompt = get_account_agent_prompt()
    clean_messages = [acc_prompt] + [m for m in messages if not isinstance(m, SystemMessage)]
    response = account_model.invoke(clean_messages)
    return {"messages": [response]}

def ticket_agent_node(state: AgentState):
    """Ticket Specialist Agent: executes ticket management, logging, customer history, and escalations using get_ticket_agent_prompt()."""
    messages = state["messages"]
    ticket_id = state.get("ticket_metadata", {}).get("ticket_id", "default_thread")
    log_event(ticket_id=ticket_id, event_type="AGENT_EXECUTION", agent_name="ticket_agent")
    
    tick_prompt = get_ticket_agent_prompt()
    clean_messages = [tick_prompt] + [m for m in messages if not isinstance(m, SystemMessage)]
    response = ticket_model.invoke(clean_messages)
    return {"messages": [response]}

# 6. Routing Functions
def supervisor_router(state: AgentState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return END
    
    last_msg = messages[-1]
    ticket_id = state.get("ticket_metadata", {}).get("ticket_id", "default_thread")
    
    # If supervisor produced a final AI message without tool calls, end
    if isinstance(last_msg, AIMessage):
        if getattr(last_msg, "tool_calls", None):
            log_event(ticket_id=ticket_id, event_type="ROUTING", agent_name="supervisor_agent", details={"destination": "tools"})
            return "tools"
        log_event(ticket_id=ticket_id, event_type="RESOLUTION", agent_name="supervisor_agent", details={"status": "resolved"})
        return END

    # Extract latest human message and state metadata for classification
    user_text = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_text = m.content
            break
            
    metadata = state.get("ticket_metadata", {})
    classification = classify_ticket(user_text, metadata)
    
    log_event(
        ticket_id=ticket_id,
        event_type="CLASSIFICATION",
        agent_name="supervisor_agent",
        details=classification
    )
    log_event(
        ticket_id=ticket_id,
        event_type="ROUTING",
        agent_name="supervisor_agent",
        details={"destination": classification["selected_agent"], "reason": classification["routing_reason"]}
    )
    
    return classification["selected_agent"]

def specialist_router(state: AgentState) -> str:
    messages = state.get("messages", [])
    ticket_id = state.get("ticket_metadata", {}).get("ticket_id", "default_thread")
    if not messages:
        return END
    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
        for tc in last_msg.tool_calls:
            log_event(
                ticket_id=ticket_id,
                event_type="TOOL_CALL",
                tool_name=tc.get("name"),
                details={"args": tc.get("args")}
            )
        return "tools"
    
    log_event(ticket_id=ticket_id, event_type="RESOLUTION", details={"status": "resolved"})
    return END

def tool_router(state: AgentState) -> str:
    """Routes tool response back to the appropriate agent or END."""
    messages = state.get("messages", [])
    ticket_id = state.get("ticket_metadata", {}).get("ticket_id", "default_thread")
    if not messages:
        return END
    last_msg = messages[-1]
    if isinstance(last_msg, ToolMessage):
        tool_name = getattr(last_msg, "name", "") or ""
        content = last_msg.content
        outcome = "error" if "error" in str(content).lower() else "success"
        log_event(
            ticket_id=ticket_id,
            event_type="TOOL_RESULT",
            tool_name=tool_name,
            outcome=outcome,
            details={"content_preview": str(content)[:200]}
        )
        if tool_name in [t.name for t in support_tools]:
            return "support_agent"
        elif tool_name in [t.name for t in account_tools]:
            return "account_agent"
        elif tool_name in [t.name for t in ticket_tools]:
            return "ticket_agent"
            
    return "supervisor_agent"

# 7. Build Multi-Agent StateGraph
builder = StateGraph(AgentState)

# Add 4 Specialized Executable Nodes + ToolNode
builder.add_node("supervisor_agent", supervisor_agent_node)
builder.add_node("support_agent", support_agent_node)
builder.add_node("account_agent", account_agent_node)
builder.add_node("ticket_agent", ticket_agent_node)
builder.add_node("tools", ToolNode(all_tools))

# Add Edges
builder.add_edge(START, "supervisor_agent")
builder.add_conditional_edges(
    "supervisor_agent",
    supervisor_router,
    {
        "support_agent": "support_agent",
        "account_agent": "account_agent",
        "ticket_agent": "ticket_agent",
        "tools": "tools",
        END: END,
    }
)
builder.add_conditional_edges(
    "support_agent",
    specialist_router,
    {
        "tools": "tools",
        END: END,
    }
)
builder.add_conditional_edges(
    "account_agent",
    specialist_router,
    {
        "tools": "tools",
        END: END,
    }
)
builder.add_conditional_edges(
    "ticket_agent",
    specialist_router,
    {
        "tools": "tools",
        END: END,
    }
)
builder.add_conditional_edges(
    "tools",
    tool_router,
    {
        "support_agent": "support_agent",
        "account_agent": "account_agent",
        "ticket_agent": "ticket_agent",
        "supervisor_agent": "supervisor_agent",
        END: END,
    }
)

# Compile Graph with MemorySaver checkpointer
checkpointer = MemorySaver()
orchestrator = builder.compile(checkpointer=checkpointer)