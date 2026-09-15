import pytest
from unittest.mock import patch
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from agentic.workflow import orchestrator

def test_workflow_graph_structure():
    """Tests that the orchestrator graph is properly compiled with memory checkpointer."""
    assert orchestrator is not None
    assert hasattr(orchestrator, "checkpointer")
    assert orchestrator.checkpointer is not None

def test_workflow_execution_with_mock():
    """Tests orchestrator flow execution using a mock LLM response."""
    mock_msg = AIMessage(content="Your CultPass subscription includes 4 curated experiences per month.")
    mock_result = ChatResult(generations=[ChatGeneration(message=mock_msg)])
    
    with patch("langchain_openai.ChatOpenAI._generate", return_value=mock_result):
        config = {"configurable": {"thread_id": "test_thread_mock"}}
        input_data = {"messages": [HumanMessage(content="What is included in CultPass subscription?")]}
        
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
