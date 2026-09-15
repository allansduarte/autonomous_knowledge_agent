import pytest
from unittest.mock import patch
from agentic.tools import (
    get_user_profile,
    get_user_reservations,
    search_experiences,
    get_ticket_details,
    update_ticket_status,
    log_ticket_message,
    get_customer_history,
    escalate_ticket,
    search_knowledge_base,
)

def test_cultpass_user_profile():
    profile = get_user_profile.invoke({"user_id_or_email": "a4ab87"})
    assert "user_id" in profile
    assert profile["user_id"] == "a4ab87"
    assert "subscription" in profile

def test_cultpass_user_reservations():
    reservations = get_user_reservations.invoke({"user_id": "a4ab87"})
    assert isinstance(reservations, list)

def test_search_experiences():
    experiences = search_experiences.invoke({"query": "Samba"})
    assert isinstance(experiences, list)
    assert len(experiences) > 0

def test_rag_knowledge_search_with_confidence_and_escalation():
    # 1. High confidence query
    res = search_knowledge_base.invoke({"query": "how to cancel or pause subscription", "top_k": 3})
    assert isinstance(res, dict)
    assert "highest_confidence" in res
    assert res["should_escalate"] is False
    assert len(res["articles"]) > 0
    
    # 2. Unknown/Low confidence query triggering escalation suggestion
    res_low = search_knowledge_base.invoke({"query": "xyz123999 unknown quantum feature", "top_k": 3})
    assert res_low["should_escalate"] is True
    assert res_low["escalation_reason"] is not None

def test_udahub_ticket_details_and_update():
    ticket_id = "ticket_a4ab87"
    details = get_ticket_details.invoke({"ticket_id": ticket_id})
    assert "ticket_id" in details
    
    update_res = update_ticket_status.invoke({
        "ticket_id": ticket_id,
        "status": "pending",
        "issue_type": "cancellation",
        "tags": "cancellation, billing"
    })
    assert update_res["success"] is True
    
    log_res = log_ticket_message.invoke({
        "ticket_id": ticket_id,
        "role": "user",
        "content": "Automated unit test message."
    })
    assert log_res["success"] is True

def test_get_customer_history_and_escalate_ticket():
    history = get_customer_history.invoke({"external_user_id": "a4ab87"})
    assert history["user_found"] is True
    assert "ticket_history" in history
    
    esc = escalate_ticket.invoke({"ticket_id": "ticket_a4ab87", "reason": "Complex billing dispute"})
    assert esc["success"] is True
    assert esc["status"] == "escalated"

def test_validation_and_error_handling_empty_identifiers():
    """Tests validation when empty user or ticket identifiers are provided."""
    res1 = get_user_profile.invoke({"user_id_or_email": ""})
    assert "error" in res1
    assert "cannot be empty" in res1["error"]
    
    res2 = get_user_reservations.invoke({"user_id": "   "})
    assert "error" in res2
    assert "cannot be empty" in res2["error"]

    res3 = get_ticket_details.invoke({"ticket_id": ""})
    assert "error" in res3
    assert "cannot be empty" in res3["error"]

def test_unknown_account_vs_empty_reservations():
    """Tests distinguishing unknown accounts from valid accounts with zero reservations."""
    # Non-existent user
    res_unknown = get_user_reservations.invoke({"user_id": "nonexistent_user_999"})
    assert isinstance(res_unknown, dict)
    assert "error" in res_unknown
    assert "not found" in res_unknown["error"]

def test_database_exception_handling():
    """Tests catching database operational exceptions in tools."""
    with patch("agentic.tools.cultpass_db_tool.get_cultpass_engine", side_effect=Exception("Database connection lost")):
        res = get_user_profile.invoke({"user_id_or_email": "a4ab87"})
        assert "error" in res
        assert "Database error" in res["error"]
