import pytest
from unittest.mock import patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from agentic.workflow import orchestrator, classify_ticket, supervisor_router
from agentic.logger import get_ticket_events, get_metrics_summary, clear_logs
from agentic.tools import (
    search_knowledge_base,
    get_user_profile,
    get_user_reservations,
    escalate_ticket,
    update_ticket_status
)

def test_workflow_graph_structure():
    """Tests that the orchestrator graph is properly compiled with memory checkpointer and contains all 4 specialized agent nodes."""
    assert orchestrator is not None
    assert hasattr(orchestrator, "checkpointer")
    assert orchestrator.checkpointer is not None
    
    # Verify graph contains all 4 specialized agent nodes + tools
    nodes = orchestrator.nodes
    assert "supervisor_agent" in nodes
    assert "support_agent" in nodes
    assert "account_agent" in nodes
    assert "ticket_agent" in nodes
    assert "tools" in nodes

def test_ticket_classification_policy_sample():
    """Tests classification and routing for policy / FAQ tickets."""
    msg = "How do I cancel or pause my CultPass subscription?"
    metadata = {"tags": "cancellation, policy", "urgency": "normal"}
    res = classify_ticket(msg, metadata)
    
    assert res["selected_agent"] == "support_agent"
    assert res["category"] == "policy"
    assert res["urgency"] == "normal"
    assert "Knowledge Support Agent" in res["routing_reason"]

def test_ticket_classification_account_sample():
    """Tests classification and routing for user profile, quota, and reservation tickets."""
    msg = "What is my remaining pass quota and current reservation list for user a4ab87?"
    metadata = {"tags": "account, quota", "urgency": "normal"}
    res = classify_ticket(msg, metadata)
    
    assert res["selected_agent"] == "account_agent"
    assert res["category"] == "account"
    assert "Account Agent" in res["routing_reason"]

def test_ticket_classification_complex_escalation_sample():
    """Tests classification and routing for complex billing disputes and high-urgency tickets."""
    msg = "I have an unrecognized charge on my credit card and need an immediate escalation to a manager."
    metadata = {"tags": "billing, dispute", "urgency": "high", "status": "open"}
    res = classify_ticket(msg, metadata)
    
    assert res["selected_agent"] == "ticket_agent"
    assert res["category"] == "complex"
    assert res["urgency"] == "high"
    assert "Ticket Agent" in res["routing_reason"]

def test_supervisor_router_with_metadata():
    """Tests supervisor_router selecting the target specialist node based on message and metadata state."""
    state = {
        "messages": [HumanMessage(content="Please check ticket history for customer dispute")],
        "ticket_metadata": {"tags": "dispute", "urgency": "high"}
    }
    target_node = supervisor_router(state)
    assert target_node == "ticket_agent"

def test_end_to_end_scenario1_policy_successful_resolution():
    """Scenario 1: Policy / FAQ query -> support_agent -> RAG tool -> successful resolution in DB."""
    ticket_id = "test_ticket_policy_001"
    
    # Execute RAG knowledge retrieval tool
    kb_res = search_knowledge_base.invoke({"query": "how to cancel or pause subscription", "top_k": 3})
    assert len(kb_res["articles"]) > 0
    assert kb_res["should_escalate"] is False
    
    # Update DB ticket status to resolved
    upd_res = update_ticket_status.invoke({"ticket_id": ticket_id, "status": "resolved", "issue_type": "cancellation"})
    assert upd_res["success"] is True
    assert upd_res["status"] == "resolved"
    
    # Verify structured logs
    events = get_ticket_events(ticket_id)
    assert isinstance(events, list)

def test_end_to_end_scenario2_unavailable_knowledge_escalation():
    """Scenario 2: Unavailable knowledge / low confidence query -> should_escalate: True -> escalate_ticket in DB."""
    ticket_id = "test_ticket_escalation_002"
    
    # Execute low-confidence search
    kb_res = search_knowledge_base.invoke({"query": "unknown quantum portal feature 999", "top_k": 3})
    assert kb_res["should_escalate"] is True
    
    # Execute human escalation tool
    esc_res = escalate_ticket.invoke({"ticket_id": ticket_id, "reason": kb_res["escalation_reason"]})
    assert esc_res["success"] is True
    assert esc_res["status"] == "escalated"

def test_end_to_end_scenario3_account_services_resolution():
    """Scenario 3: Account lookup query -> account_agent -> get_user_profile & get_user_reservations -> resolved in DB."""
    ticket_id = "test_ticket_account_003"
    
    # Execute account tools
    profile = get_user_profile.invoke({"user_id_or_email": "a4ab87"})
    assert "user_id" in profile
    assert profile["user_id"] == "a4ab87"
    
    reservations = get_user_reservations.invoke({"user_id": "a4ab87"})
    assert isinstance(reservations, list)
    
    upd_res = update_ticket_status.invoke({"ticket_id": ticket_id, "status": "resolved", "issue_type": "account_inquiry"})
    assert upd_res["success"] is True

def test_end_to_end_scenario4_edge_case_and_error_handling():
    """Scenario 4: Edge case / unknown user lookup -> handled gracefully with error dict."""
    ticket_id = "test_ticket_edge_004"
    
    res_unknown = get_user_reservations.invoke({"user_id": "nonexistent_user_xyz999"})
    assert "error" in res_unknown
    assert "not found" in res_unknown["error"]

def test_structured_operational_metrics_summary():
    """Tests structured operational metric calculation."""
    metrics = get_metrics_summary()
    assert isinstance(metrics, dict)
    assert "total_events" in metrics
    assert "retrieval_success_rate" in metrics

def test_workflow_execution_with_mock():
    """Tests orchestrator flow execution using a mock LLM response."""
    mock_msg = AIMessage(content="Your CultPass subscription includes 4 curated experiences per month.")
    mock_result = ChatResult(generations=[ChatGeneration(message=mock_msg)])
    
    with patch("langchain_openai.ChatOpenAI._generate", return_value=mock_result):
        config = {"configurable": {"thread_id": "test_thread_mock"}}
        input_data = {
            "messages": [HumanMessage(content="What is included in CultPass subscription?")],
            "ticket_metadata": {"ticket_id": "test_thread_mock"}
        }
        
        result = orchestrator.invoke(input=input_data, config=config)
        assert "messages" in result
        assert len(result["messages"]) > 0
        last_msg = result["messages"][-1]
        assert "4 curated experiences" in last_msg.content

def test_workflow_memory_persistence():
    """Tests short-term thread session memory checkpointing in state graph."""
    mock_msg1 = AIMessage(content="You have 4 pass credits monthly.")
    mock_msg2 = AIMessage(content="As mentioned, 4 passes are included.")
    
    res1 = ChatResult(generations=[ChatGeneration(message=mock_msg1)])
    res2 = ChatResult(generations=[ChatGeneration(message=mock_msg2)])
    
    with patch("langchain_openai.ChatOpenAI._generate") as mock_gen:
        mock_gen.side_effect = [res1, res2]
        
        config = {"configurable": {"thread_id": "test_thread_persistence"}}
        
        # Turn 1
        orchestrator.invoke({"messages": [HumanMessage(content="How many credits?")]}, config=config)
        
        # Turn 2
        orchestrator.invoke({"messages": [HumanMessage(content="Can you repeat that?")]}, config=config)
        
        history = list(orchestrator.get_state_history(config=config))
        assert len(history) > 0
        messages = history[0].values["messages"]
        assert len(messages) >= 4  # Includes user inputs and responses across turns
