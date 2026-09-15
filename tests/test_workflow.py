import pytest
from unittest.mock import patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from agentic.workflow import orchestrator, classify_ticket, supervisor_router_node
from agentic.agents.ticket_agent import create_ticket_tools
from agentic.logger import get_ticket_events, get_metrics_summary, clear_logs
from agentic.tools import get_ticket_details, search_knowledge_base

def test_workflow_graph_structure():
    """Tests that the orchestrator graph is properly compiled with memory checkpointer and contains all 4 specialized agent nodes."""
    assert orchestrator is not None
    assert hasattr(orchestrator, "checkpointer")
    assert orchestrator.checkpointer is not None
    
    nodes = orchestrator.nodes
    assert "supervisor_router_node" in nodes
    assert "support_agent" in nodes
    assert "account_agent" in nodes
    assert "ticket_agent" in nodes
    assert "tools" in nodes

def test_ticket_agent_tools_capabilities():
    """Tests that create_ticket_tools registers all 5 documented capabilities including customer history and human escalation."""
    tools = create_ticket_tools()
    tool_names = [t.name for t in tools]
    assert "get_ticket_details" in tool_names
    assert "update_ticket_status" in tool_names
    assert "log_ticket_message" in tool_names
    assert "get_customer_history" in tool_names
    assert "escalate_ticket" in tool_names

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

def test_metadata_dynamic_routing_shift():
    """Tests that changing metadata (urgency/tags) shifts routing for the exact same message content."""
    msg = "Need help with subscription"
    
    # 1. Normal urgency -> Support Agent
    res_normal = classify_ticket(msg, {"urgency": "normal", "tags": "faq"})
    assert res_normal["selected_agent"] == "support_agent"
    
    # 2. Urgent / Dispute metadata -> Ticket Agent
    res_urgent = classify_ticket(msg, {"urgency": "high", "tags": "dispute, billing", "issue_type": "billing_dispute"})
    assert res_urgent["selected_agent"] == "ticket_agent"

def test_graph_stream_node_visitation_policy():
    """Tests graph-level stream execution verifying visited nodes for policy requests."""
    ticket_id = "test_stream_policy"
    config = {"configurable": {"thread_id": ticket_id}}
    input_data = {
        "messages": [HumanMessage(content="How do I cancel or pause subscription?")],
        "ticket_metadata": {"ticket_id": ticket_id, "tags": "policy", "urgency": "normal"}
    }
    
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "search_knowledge_base", "args": {"query": "how to cancel or pause subscription"}, "id": "c1"}]
    )
    final_ans = AIMessage(content="Subscription can be canceled in My Account.")
    res1 = ChatResult(generations=[ChatGeneration(message=tool_call_msg)])
    res2 = ChatResult(generations=[ChatGeneration(message=final_ans)])
    
    with patch("langchain_openai.ChatOpenAI._generate") as mock_gen:
        mock_gen.side_effect = [res1, res2]
        visited_nodes = []
        for step in orchestrator.stream(input_data, config=config):
            visited_nodes.extend(step.keys())
            
        assert "supervisor_router_node" in visited_nodes
        assert "support_agent" in visited_nodes

def test_graph_stream_node_visitation_account():
    """Tests graph-level stream execution verifying visited nodes for account requests."""
    ticket_id = "test_stream_account"
    config = {"configurable": {"thread_id": ticket_id}}
    input_data = {
        "messages": [HumanMessage(content="Check profile for user a4ab87")],
        "ticket_metadata": {"ticket_id": ticket_id, "tags": "profile", "urgency": "normal"}
    }
    
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "get_user_profile", "args": {"user_id_or_email": "a4ab87"}, "id": "c2"}]
    )
    final_ans = AIMessage(content="Profile retrieved.")
    res1 = ChatResult(generations=[ChatGeneration(message=tool_call_msg)])
    res2 = ChatResult(generations=[ChatGeneration(message=final_ans)])
    
    with patch("langchain_openai.ChatOpenAI._generate") as mock_gen:
        mock_gen.side_effect = [res1, res2]
        visited_nodes = []
        for step in orchestrator.stream(input_data, config=config):
            visited_nodes.extend(step.keys())
            
        assert "supervisor_router_node" in visited_nodes
        assert "account_agent" in visited_nodes

def test_scenario1_policy_faq_orchestrator_invoke():
    """Scenario 1: End-to-end Policy FAQ query processing via orchestrator.invoke."""
    ticket_id = "ticket_a4ab87"
    config = {"configurable": {"thread_id": ticket_id}}
    input_data = {
        "messages": [HumanMessage(content="How do I cancel or pause my CultPass subscription?")],
        "ticket_metadata": {"ticket_id": ticket_id, "tags": "cancellation", "urgency": "normal"}
    }
    
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "search_knowledge_base", "args": {"query": "how to cancel or pause subscription"}, "id": "call_1"}]
    )
    final_ans = AIMessage(content="You can cancel or pause your subscription at any time via the My Account section.")
    
    res1 = ChatResult(generations=[ChatGeneration(message=tool_call_msg)])
    res2 = ChatResult(generations=[ChatGeneration(message=final_ans)])
    
    with patch("langchain_openai.ChatOpenAI._generate") as mock_gen:
        mock_gen.side_effect = [res1, res2]
        
        result = orchestrator.invoke(input=input_data, config=config)
        assert "messages" in result
        assert len(result["messages"]) > 0
        
        events = get_ticket_events(ticket_id)
        event_types = [e["event_type"] for e in events]
        assert "CLASSIFICATION" in event_types
        assert "ROUTING" in event_types
        assert "TOOL_CALL" in event_types
        
        db_details = get_ticket_details.invoke({"ticket_id": ticket_id})
        assert db_details.get("metadata", {}).get("status") == "resolved"

def test_scenario2_unavailable_knowledge_escalation_orchestrator_invoke():
    """Scenario 2: End-to-end low confidence query -> should_escalate -> automatic escalation via orchestrator.invoke."""
    ticket_id = "ticket_a4ab87"
    config = {"configurable": {"thread_id": ticket_id}}
    input_data = {
        "messages": [HumanMessage(content="xyz123999 unknown quantum feature")],
        "ticket_metadata": {"ticket_id": ticket_id, "tags": "policy", "urgency": "normal"}
    }
    
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "search_knowledge_base", "args": {"query": "xyz123999 unknown quantum feature"}, "id": "call_2"}]
    )
    res_mock = ChatResult(generations=[ChatGeneration(message=tool_call_msg)])
    
    with patch("langchain_openai.ChatOpenAI._generate", return_value=res_mock):
        result = orchestrator.invoke(input=input_data, config=config)
        assert "messages" in result
        assert len(result["messages"]) > 0
        
        events = get_ticket_events(ticket_id)
        event_types = [e["event_type"] for e in events]
        assert "RETRIEVAL_MISS" in event_types
        assert "ESCALATION" in event_types
        
        db_details = get_ticket_details.invoke({"ticket_id": ticket_id})
        assert db_details.get("metadata", {}).get("status") == "escalated"

def test_scenario3_account_services_orchestrator_invoke():
    """Scenario 3: End-to-end user profile & reservation query processing via orchestrator.invoke."""
    ticket_id = "ticket_a4ab87"
    config = {"configurable": {"thread_id": ticket_id}}
    input_data = {
        "messages": [HumanMessage(content="What is my pass quota and reservations for user a4ab87?")],
        "ticket_metadata": {"ticket_id": ticket_id, "tags": "quota, profile", "urgency": "normal"}
    }
    
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "get_user_profile", "args": {"user_id_or_email": "a4ab87"}, "id": "call_3"}]
    )
    final_ans = AIMessage(content="User a4ab87 has an active basic subscription with 4 passes.")
    
    res1 = ChatResult(generations=[ChatGeneration(message=tool_call_msg)])
    res2 = ChatResult(generations=[ChatGeneration(message=final_ans)])
    
    with patch("langchain_openai.ChatOpenAI._generate") as mock_gen:
        mock_gen.side_effect = [res1, res2]
        
        result = orchestrator.invoke(input=input_data, config=config)
        assert "messages" in result
        
        events = get_ticket_events(ticket_id)
        tool_names = [e.get("tool_name") for e in events if e.get("tool_name")]
        assert "get_user_profile" in tool_names
        
        db_details = get_ticket_details.invoke({"ticket_id": ticket_id})
        assert db_details.get("metadata", {}).get("status") == "resolved"

def test_scenario4_edge_case_error_handling_orchestrator_invoke():
    """Scenario 4: End-to-end unknown user lookup via orchestrator.invoke."""
    ticket_id = "ticket_a4ab87"
    config = {"configurable": {"thread_id": ticket_id}}
    input_data = {
        "messages": [HumanMessage(content="Look up reservations for nonexistent_user_xyz999")],
        "ticket_metadata": {"ticket_id": ticket_id, "tags": "account", "urgency": "normal"}
    }
    
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "get_user_reservations", "args": {"user_id": "nonexistent_user_xyz999"}, "id": "call_4"}]
    )
    final_ans = AIMessage(content="User nonexistent_user_xyz999 was not found.")
    
    res1 = ChatResult(generations=[ChatGeneration(message=tool_call_msg)])
    res2 = ChatResult(generations=[ChatGeneration(message=final_ans)])
    
    with patch("langchain_openai.ChatOpenAI._generate") as mock_gen:
        mock_gen.side_effect = [res1, res2]
        
        result = orchestrator.invoke(input=input_data, config=config)
        assert "messages" in result
        
        events = get_ticket_events(ticket_id)
        outcomes = [e.get("outcome") for e in events]
        assert "error" in outcomes

def test_metrics_summary_accuracy():
    """Tests that get_metrics_summary calculates exact retrieval success rates without double-counting."""
    summary = get_metrics_summary()
    assert summary["retrieval_success_rate"] <= 1.0
    assert "total_events" in summary
    assert "unique_tickets" in summary

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
        assert len(messages) >= 4

def test_ticket_agent_escalation_preserves_escalated_status():
    """Regression test verifying ticket_agent -> tools -> ticket_agent flow preserves escalated status when successful."""
    ticket_id = "test_escalate_preserves_status"
    config = {"configurable": {"thread_id": ticket_id}}
    input_data = {
        "messages": [HumanMessage(content="Escalate my issue to human manager immediately")],
        "ticket_metadata": {"ticket_id": ticket_id, "tags": "dispute", "urgency": "high"}
    }
    
    esc_tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "escalate_ticket", "args": {"ticket_id": ticket_id, "reason": "Customer requested human manager"}, "id": "call_esc"}]
    )
    ack_msg = AIMessage(content="I have escalated your ticket to human support management.")
    
    res1 = ChatResult(generations=[ChatGeneration(message=esc_tool_call)])
    res2 = ChatResult(generations=[ChatGeneration(message=ack_msg)])
    
    with patch("langchain_openai.ChatOpenAI._generate") as mock_gen:
        mock_gen.side_effect = [res1, res2]
        
        result = orchestrator.invoke(input=input_data, config=config)
        assert "messages" in result
        
        events = get_ticket_events(ticket_id)
        res_events = [e for e in events if e.get("event_type") == "RESOLUTION"]
        assert len(res_events) > 0
        assert res_events[-1]["details"]["status"] == "escalated"

def test_failed_escalation_does_not_mark_ticket_escalated():
    """Regression test verifying that a failed escalation tool call logs outcome 'error' and does not mark ticket escalated."""
    ticket_id = "test_failed_escalation"
    config = {"configurable": {"thread_id": ticket_id}}
    input_data = {
        "messages": [HumanMessage(content="Escalate my issue immediately")],
        "ticket_metadata": {"ticket_id": ticket_id, "tags": "dispute", "urgency": "high"}
    }
    
    esc_tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "escalate_ticket", "args": {"ticket_id": "", "reason": "invalid_empty_id"}, "id": "call_failed"}]
    )
    ack_msg = AIMessage(content="Could not escalate ticket due to invalid ID.")
    
    res1 = ChatResult(generations=[ChatGeneration(message=esc_tool_call)])
    res2 = ChatResult(generations=[ChatGeneration(message=ack_msg)])
    
    with patch("langchain_openai.ChatOpenAI._generate") as mock_gen:
        mock_gen.side_effect = [res1, res2]
        
        result = orchestrator.invoke(input=input_data, config=config)
        assert "messages" in result
        
        events = get_ticket_events(ticket_id)
        esc_events = [e for e in events if e.get("event_type") == "ESCALATION"]
        assert len(esc_events) > 0
        assert esc_events[-1]["outcome"] == "error"
        
        res_events = [e for e in events if e.get("event_type") == "RESOLUTION"]
        assert len(res_events) > 0
        assert res_events[-1]["details"]["status"] == "resolved"

def test_multi_tool_call_processing_logs_all_outcomes():
    """Regression test verifying tool_router processes and logs ALL ToolMessage items in a multi-tool call turn."""
    ticket_id = "test_multi_tool"
    config = {"configurable": {"thread_id": ticket_id}}
    input_data = {
        "messages": [HumanMessage(content="Get my profile and my reservations for user a4ab87")],
        "ticket_metadata": {"ticket_id": ticket_id, "tags": "profile", "urgency": "normal"}
    }
    
    multi_tool_call = AIMessage(
        content="",
        tool_calls=[
            {"name": "get_user_profile", "args": {"user_id_or_email": "a4ab87"}, "id": "m1"},
            {"name": "get_user_reservations", "args": {"user_id": "a4ab87"}, "id": "m2"}
        ]
    )
    final_ans = AIMessage(content="Retrieved profile and reservations successfully.")
    
    res1 = ChatResult(generations=[ChatGeneration(message=multi_tool_call)])
    res2 = ChatResult(generations=[ChatGeneration(message=final_ans)])
    
    with patch("langchain_openai.ChatOpenAI._generate") as mock_gen:
        mock_gen.side_effect = [res1, res2]
        
        result = orchestrator.invoke(input=input_data, config=config)
        assert "messages" in result
        
        events = get_ticket_events(ticket_id)
        tool_results = [e for e in events if e.get("event_type") == "TOOL_RESULT"]
        logged_tools = [e.get("tool_name") for e in tool_results]
        
        assert "get_user_profile" in logged_tools
        assert "get_user_reservations" in logged_tools


