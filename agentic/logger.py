import datetime
from typing import Dict, Any, List, Optional

# In-memory structured event store keyed by ticket_id
_EVENT_LOGS: List[Dict[str, Any]] = []

def log_event(
    ticket_id: str,
    event_type: str,
    agent_name: Optional[str] = None,
    tool_name: Optional[str] = None,
    outcome: Optional[str] = "success",
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Logs a structured event linked by ticket_id including timestamps, event_type, tool_name, and details."""
    event = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "ticket_id": str(ticket_id),
        "event_type": str(event_type).upper(),
        "agent_name": agent_name,
        "tool_name": tool_name,
        "outcome": outcome,
        "details": details or {}
    }
    _EVENT_LOGS.append(event)
    return event

def get_ticket_events(ticket_id: str) -> List[Dict[str, Any]]:
    """Retrieves all structured events for a given ticket_id."""
    return [e for e in _EVENT_LOGS if e["ticket_id"] == str(ticket_id)]

def get_all_events() -> List[Dict[str, Any]]:
    """Retrieves all recorded event logs."""
    return list(_EVENT_LOGS)

def clear_logs() -> None:
    """Clears in-memory event logs (useful between test runs)."""
    global _EVENT_LOGS
    _EVENT_LOGS = []

def get_metrics_summary() -> Dict[str, Any]:
    """Summarizes retrieval success, escalation frequency, and tool usage across all recorded event logs."""
    total_events = len(_EVENT_LOGS)
    ticket_ids = set(e["ticket_id"] for e in _EVENT_LOGS)
    
    tool_calls = [e for e in _EVENT_LOGS if e["event_type"] == "TOOL_CALL"]
    retrieval_calls = [e for e in tool_calls if e.get("tool_name") == "search_knowledge_base"]
    retrieval_successes = [e for e in _EVENT_LOGS if e["event_type"] == "TOOL_RESULT" and e.get("tool_name") == "search_knowledge_base" and e.get("outcome") == "success"]
    
    escalations = [e for e in _EVENT_LOGS if e["event_type"] in ["ESCALATION", "TOOL_CALL"] and (e.get("tool_name") == "escalate_ticket" or e.get("event_type") == "ESCALATION")]
    
    tool_usage_counts = {}
    for tc in tool_calls:
        tname = tc.get("tool_name", "unknown")
        tool_usage_counts[tname] = tool_usage_counts.get(tname, 0) + 1
        
    return {
        "total_events": total_events,
        "unique_tickets": len(ticket_ids),
        "total_tool_calls": len(tool_calls),
        "retrieval_calls": len(retrieval_calls),
        "retrieval_success_rate": (len(retrieval_successes) / len(retrieval_calls)) if retrieval_calls else 1.0,
        "escalation_count": len(escalations),
        "tool_usage_counts": tool_usage_counts
    }
