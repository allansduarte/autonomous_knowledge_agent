from typing import Annotated, Sequence, TypedDict, Optional, Dict, Any
import os
import json
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
from agentic.tools import escalate_ticket, update_ticket_status

load_dotenv()

# 1. State Schema
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    ticket_metadata: Optional[Dict[str, Any]]
    selected_agent: Optional[str]

# 2. Specialist Tools Registration
support_tools = create_support_tools() + [escalate_ticket]
account_tools = create_account_tools()
ticket_tools = create_ticket_tools()

all_tools = []
for t in support_tools + account_tools + ticket_tools:
    if t not in all_tools:
        all_tools.append(t)

# 3. Models bound with respective tools
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY") or "sk-dummy-key-for-init"
base_model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

support_model = base_model.bind_tools(support_tools)
account_model = base_model.bind_tools(account_tools)
ticket_model = base_model.bind_tools(ticket_tools)

# 4. Scored Multi-Feature Ticket Classification Function
def classify_ticket(message_text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Classifies incoming tickets based on multi-feature scoring: content keywords, semantic intent, and metadata flags.
    Returns a dictionary containing selected_agent, category, urgency, routing_reason, and component scores.
    """
    metadata = metadata or {}
    text_lower = (message_text or "").lower()
    tags = str(metadata.get("tags", "")).lower()
    urgency = str(metadata.get("urgency", "normal")).lower()
    status = str(metadata.get("status", "")).lower()
    issue_type = str(metadata.get("issue_type", "")).lower()
    
    complex_score = 0.0
    account_score = 0.0
    policy_score = 0.0
    
    # --- Metadata Evaluation ---
    if urgency in ["high", "urgent", "critical"]:
        complex_score += 4.0
    if any(k in tags for k in ["billing", "escalated", "dispute", "fraud", "complaint"]):
        complex_score += 3.5
    if issue_type in ["billing_dispute", "account_takeover", "escalation"]:
        complex_score += 4.0
    if status in ["escalated", "blocked"]:
        complex_score += 3.0
        
    if any(k in tags for k in ["profile", "reservation", "account", "quota", "pass"]):
        account_score += 3.0
    if any(k in tags for k in ["cancellation", "policy", "faq", "rules", "refund"]):
        policy_score += 3.0

    # --- Text Content Keyword & Intent Evaluation ---
    ticket_keywords = ["ticket", "status", "history", "log", "escalat", "dispute", "urgent", "human", "blocked account", "charge", "manager", "unrecognized"]
    account_keywords = ["profile", "reservation", "quota", "experience", "tier", "user_id", "subscription status", "my pass", "book", "event catalog"]
    policy_keywords = ["cancel", "pause", "policy", "faq", "rules", "how do i", "how to", "terms", "condition", "refund policy"]
    
    for kw in ticket_keywords:
        if kw in text_lower:
            complex_score += 2.0
    for kw in account_keywords:
        if kw in text_lower:
            account_score += 2.0
    for kw in policy_keywords:
        if kw in text_lower:
            policy_score += 2.0
            
    # Default baseline score for policy lookup
    policy_score += 1.0

    # Score comparison and selected destination determination
    if complex_score > account_score and complex_score > policy_score:
        selected = "ticket_agent"
        category = "complex"
        calc_urgency = "high" if urgency in ["high", "urgent", "critical"] or complex_score >= 4.0 else "medium"
        reason = f"Routed to Ticket Agent based on high complex/escalation score ({complex_score:.1f}) driven by urgency, metadata, or dispute keywords."
    elif account_score >= policy_score:
        selected = "account_agent"
        category = "account"
        calc_urgency = "normal"
        reason = f"Routed to Account Agent based on account/reservation query score ({account_score:.1f})."
    else:
        selected = "support_agent"
        category = "policy"
        calc_urgency = "normal"
        reason = f"Routed to Knowledge Support Agent based on policy/FAQ score ({policy_score:.1f})."

    return {
        "selected_agent": selected,
        "category": category,
        "urgency": calc_urgency,
        "routing_reason": reason,
        "scores": {
            "complex": complex_score,
            "account": account_score,
            "policy": policy_score
        }
    }

# 5. Agent Executable Node Functions
def supervisor_router_node(state: AgentState) -> Dict[str, Any]:
    """Dedicated Supervisor Router Node at graph START.
    Runs ticket classification BEFORE model execution, logs routing decisions, and sets selected_agent state.
    """
    messages = state.get("messages", [])
    ticket_id = state.get("ticket_metadata", {}).get("ticket_id", "default_thread")
    
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
        agent_name="supervisor_router_node",
        details=classification
    )
    log_event(
        ticket_id=ticket_id,
        event_type="ROUTING",
        agent_name="supervisor_router_node",
        details={"destination": classification["selected_agent"], "reason": classification["routing_reason"]}
    )
    
    return {"selected_agent": classification["selected_agent"]}

def dispatch_specialist(state: AgentState) -> str:
    """Conditional edge from supervisor_router_node to target specialist agent node."""
    return state.get("selected_agent") or "support_agent"

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

# 6. Specialist & Tool Routers
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
    
    # Check if escalation tool was executed during conversation
    has_escalated = False
    for m in messages:
        if isinstance(m, ToolMessage) and getattr(m, "name", "") == "escalate_ticket":
            has_escalated = True
            break
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            if any(tc.get("name") == "escalate_ticket" for tc in m.tool_calls):
                has_escalated = True
                break
                
    final_status = "escalated" if has_escalated else "resolved"
    update_ticket_status.invoke({"ticket_id": ticket_id, "status": final_status})
    log_event(ticket_id=ticket_id, event_type="RESOLUTION", details={"status": final_status})
    return END

def tool_router(state: AgentState) -> str:
    """Routes tool response back to originating specialist agent using explicit state['selected_agent'] or handles automatic escalation handoff."""
    messages = state.get("messages", [])
    ticket_id = state.get("ticket_metadata", {}).get("ticket_id", "default_thread")
    origin_agent = state.get("selected_agent") or "support_agent"
    
    if not messages:
        return END
    last_msg = messages[-1]
    if isinstance(last_msg, ToolMessage):
        tool_name = getattr(last_msg, "name", "") or ""
        content_str = str(last_msg.content)
        
        # Check for RAG knowledge retrieval outcomes
        if tool_name == "search_knowledge_base":
            if '"should_escalate": true' in content_str.lower() or "'should_escalate': true" in content_str.lower():
                log_event(
                    ticket_id=ticket_id,
                    event_type="RETRIEVAL_MISS",
                    tool_name=tool_name,
                    outcome="miss",
                    details={"reason": "Low knowledge confidence score."}
                )
                # Automatic Escalation Handoff
                esc_res = escalate_ticket.invoke({
                    "ticket_id": ticket_id,
                    "reason": "Low knowledge confidence score (no matching article found above threshold)."
                })
                log_event(
                    ticket_id=ticket_id,
                    event_type="ESCALATION",
                    tool_name="escalate_ticket",
                    outcome="escalated",
                    details=esc_res
                )
                return END
            else:
                log_event(
                    ticket_id=ticket_id,
                    event_type="RETRIEVAL_SUCCESS",
                    tool_name=tool_name,
                    outcome="success",
                    details={"content_preview": content_str[:200]}
                )
                return origin_agent

        # Log other tool outcomes
        outcome = "error" if "error" in content_str.lower() else "success"
        log_event(
            ticket_id=ticket_id,
            event_type="TOOL_RESULT",
            tool_name=tool_name,
            outcome=outcome,
            details={"content_preview": content_str[:200]}
        )
        
        return origin_agent
            
    return "supervisor_router_node"

# 7. Build Multi-Agent StateGraph
builder = StateGraph(AgentState)

# Add Executable Router & Specialist Nodes + ToolNode
builder.add_node("supervisor_router_node", supervisor_router_node)
builder.add_node("support_agent", support_agent_node)
builder.add_node("account_agent", account_agent_node)
builder.add_node("ticket_agent", ticket_agent_node)
builder.add_node("tools", ToolNode(all_tools))

# Add Graph Edges
builder.add_edge(START, "supervisor_router_node")
builder.add_conditional_edges(
    "supervisor_router_node",
    dispatch_specialist,
    {
        "support_agent": "support_agent",
        "account_agent": "account_agent",
        "ticket_agent": "ticket_agent",
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
        "supervisor_router_node": "supervisor_router_node",
        END: END,
    }
)

# Compile Graph with MemorySaver checkpointer
checkpointer = MemorySaver()
orchestrator = builder.compile(checkpointer=checkpointer)