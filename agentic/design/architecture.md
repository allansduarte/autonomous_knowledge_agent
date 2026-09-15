# Multi-Agent System Architecture & Design Specification - Uda-hub Cultpass

## 1. Executive Summary & Pattern Overview

The Uda-hub CultPass customer support system implements a **Supervisor Multi-Agent Pattern** powered by **LangGraph**. The architecture orchestrates four specialized agents to process tickets from submission to resolution or escalation.

```mermaid
graph TD
    TicketInput([Customer Ticket / Chat Message]) --> Supervisor[Supervisor Router Agent]

    subgraph LangGraph Orchestration [LangGraph StateGraph Core]
        Supervisor -->|Route: Knowledge Query| SupportAgent[Support Agent - RAG]
        Supervisor -->|Route: User Profile / Quotas| AccountAgent[Account Agent - Cultpass DB]
        Supervisor -->|Route: Ticket Status / History| TicketAgent[Ticket Agent - Udahub DB]
        
        SupportAgent --> Tools[Tools Abstraction Layer]
        AccountAgent --> Tools
        TicketAgent --> Tools
    end

    subgraph Tools [Tools Abstraction Layer]
        RAGTool[search_knowledge_base - Confidence & Escalation]
        CultpassDBTool[get_user_profile / get_user_reservations / search_experiences]
        UdahubDBTool[get_ticket_details / update_ticket_status / log_ticket_message]
        LongTermMemTool[get_customer_history - Cross-Session Memory]
        EscalationTool[escalate_ticket - Human Handoff]
    end

    subgraph Memory & Database [Persistent Storage Layer]
        CultpassDB[(cultpass.db - Users, Subscriptions, Experiences, Reservations)]
        UdahubDB[(udahub.db - Accounts, Tickets, Metadata, Messages, Knowledge Base)]
        ShortTermMem[(LangGraph MemorySaver Checkpointer - thread_id)]
    end

    RAGTool --> UdahubDB
    CultpassDBTool --> CultpassDB
    UdahubDBTool --> UdahubDB
    LongTermMemTool --> UdahubDB
    EscalationTool --> UdahubDB
    LangGraph Orchestration <--> ShortTermMem
```

---

## 2. Specialized Agent Roles & Responsibilities

The system consists of **4 specialized agents**, each with clearly defined boundaries:

1. **Supervisor Router Agent (`agentic/agents/supervisor_agent.py`)**:
   - **Role**: Entry point and central coordinator.
   - **Responsibilities**: Classifies ticket content and metadata (issue type, urgency, tags), routes execution to appropriate specialized agents/tools, and synthesizes final responses.
   - **Pattern**: Supervisor / Router Pattern.

2. **Knowledge Support Agent (`agentic/agents/support_agent.py`)**:
   - **Role**: Policy and FAQ Specialist.
   - **Responsibilities**: Queries the 15+ articles stored in `udahub.db` Knowledge table via RAG (`search_knowledge_base`). Ensures all answers are grounded in official documentation.

3. **Account & Reservation Agent (`agentic/agents/account_agent.py`)**:
   - **Role**: User Profile & Subscription Specialist.
   - **Responsibilities**: Interacts with `cultpass.db` to check user accounts (`get_user_profile`), subscription status/tier, monthly quotas, active reservations (`get_user_reservations`), and event availability (`search_experiences`).

4. **Ticket & Memory Management Agent (`agentic/agents/ticket_agent.py`)**:
   - **Role**: Audit, Memory & Escalation Manager.
   - **Responsibilities**: Updates ticket status (`update_ticket_status`), records message logs (`log_ticket_message`), retrieves cross-session customer history (`get_customer_history`), and executes human escalation (`escalate_ticket`).

---

## 3. Ticket Routing, Classification & Escalation Logic

### Classification & Routing Matrix
Incoming tickets are classified based on content and metadata:
- **General Policy / How-To / Venue Rules**: Routed to Knowledge Support Agent (`search_knowledge_base`).
- **Account Status / Quota / Reservations**: Routed to Account Agent (`get_user_profile`, `get_user_reservations`).
- **Billing Disputes / Complex Issues**: Routed to Ticket Agent for status updates and human escalation (`escalate_ticket`).

### Escalation Protocol & Confidence Scoring
- The RAG system calculates normalized confidence scores (0.0 to 1.0) for article matches.
- If the highest confidence score is below the threshold (`0.25`), `search_knowledge_base` flags `should_escalate: True`.
- The system automatically invokes `escalate_ticket`, marks the ticket status as `"escalated"`, adds an audit message in `udahub.db`, and informs the user that a support lead will intervene.

---

## 4. State & Memory Architecture

The system implements a dual-scope memory architecture:

1. **Short-Term Session Memory (`thread_id`)**:
   - Managed in-memory via LangGraph `MemorySaver` checkpointer.
   - Scoped by `thread_id` (matched to `ticket_id`). Preserves conversation history across multi-turn exchanges within the same session.

2. **Long-Term Memory (`udahub.db`)**:
   - Persists all past tickets, metadata (status, tags, main issue type), and conversation turns in `udahub.db` (`ticket_messages` table).
   - Tool `get_customer_history` retrieves historical preferences, previous resolved tickets, and interaction patterns across different sessions for returning customers.
