import os
import uuid
from typing import Dict, Any, List, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_core.tools import tool

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UDAHUB_DB_PATH = os.path.join(PROJECT_ROOT, "data", "core", "udahub.db")

def get_udahub_engine():
    return create_engine(f"sqlite:///{UDAHUB_DB_PATH}", echo=False)

@tool
def get_ticket_details(ticket_id: str) -> Dict[str, Any]:
    """Retrieves full details for a support ticket including metadata and message history."""
    from data.models.udahub import Ticket, TicketMetadata, TicketMessage
    engine = get_udahub_engine()
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        ticket = session.query(Ticket).filter(Ticket.ticket_id == ticket_id).first()
        if not ticket:
            return {"error": f"Ticket '{ticket_id}' not found."}
        
        meta = ticket.ticket_metadata
        metadata_dict = {
            "status": meta.status if meta else "unknown",
            "main_issue_type": meta.main_issue_type if meta else None,
            "tags": meta.tags if meta else "",
        }
        
        messages = [
            {
                "message_id": m.message_id,
                "role": m.role.value if hasattr(m.role, "value") else str(m.role),
                "content": m.content,
                "created_at": str(m.created_at),
            }
            for m in ticket.messages
        ]
        
        return {
            "ticket_id": ticket.ticket_id,
            "account_id": ticket.account_id,
            "user_id": ticket.user_id,
            "channel": ticket.channel,
            "metadata": metadata_dict,
            "messages": messages,
        }
    finally:
        session.close()

@tool
def update_ticket_status(ticket_id: str, status: str, issue_type: Optional[str] = None, tags: Optional[str] = None) -> Dict[str, Any]:
    """Updates status, main issue type, or tags for a support ticket in Udahub DB."""
    from data.models.udahub import TicketMetadata
    engine = get_udahub_engine()
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        meta = session.query(TicketMetadata).filter(TicketMetadata.ticket_id == ticket_id).first()
        if not meta:
            meta = TicketMetadata(ticket_id=ticket_id, status=status)
            session.add(meta)
        
        meta.status = status
        if issue_type:
            meta.main_issue_type = issue_type
        if tags:
            meta.tags = tags
        
        session.commit()
        return {
            "success": True,
            "ticket_id": ticket_id,
            "status": meta.status,
            "issue_type": meta.main_issue_type,
            "tags": meta.tags,
        }
    except Exception as e:
        session.rollback()
        return {"error": str(e)}
    finally:
        session.close()

@tool
def log_ticket_message(ticket_id: str, role: str, content: str) -> Dict[str, Any]:
    """Logs a message into the ticket conversation history in Udahub DB for persistent long-term storage."""
    from data.models.udahub import TicketMessage, RoleEnum
    engine = get_udahub_engine()
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        role_enum = RoleEnum[role.lower()] if role.lower() in RoleEnum.__members__ else RoleEnum.ai
        msg = TicketMessage(
            message_id=str(uuid.uuid4()),
            ticket_id=ticket_id,
            role=role_enum,
            content=content,
        )
        session.add(msg)
        session.commit()
        return {"success": True, "message_id": msg.message_id}
    except Exception as e:
        session.rollback()
        return {"error": str(e)}
    finally:
        session.close()

@tool
def get_customer_history(external_user_id: str) -> Dict[str, Any]:
    """Retrieves long-term interaction history, past resolved tickets, and issue patterns across past sessions for a returning customer."""
    from data.models.udahub import User, Ticket
    engine = get_udahub_engine()
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        user = session.query(User).filter(User.external_user_id == external_user_id).first()
        if not user:
            return {"user_found": False, "tickets": []}
        
        history = []
        for t in user.tickets:
            meta = t.ticket_metadata
            history.append({
                "ticket_id": t.ticket_id,
                "status": meta.status if meta else "unknown",
                "issue_type": meta.main_issue_type if meta else "unclassified",
                "tags": meta.tags if meta else "",
                "created_at": str(t.created_at),
                "message_count": len(t.messages),
                "latest_message": t.messages[-1].content if t.messages else None
            })
            
        return {
            "user_found": True,
            "user_id": user.user_id,
            "user_name": user.user_name,
            "ticket_history": history
        }
    finally:
        session.close()

@tool
def escalate_ticket(ticket_id: str, reason: str) -> Dict[str, Any]:
    """Escalates a ticket to human support lead when knowledge base confidence is low or user demands human intervention."""
    from data.models.udahub import TicketMetadata, TicketMessage, RoleEnum
    engine = get_udahub_engine()
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        meta = session.query(TicketMetadata).filter(TicketMetadata.ticket_id == ticket_id).first()
        if meta:
            meta.status = "escalated"
            meta.tags = (meta.tags + ", escalated") if meta.tags else "escalated"
        else:
            meta = TicketMetadata(ticket_id=ticket_id, status="escalated", tags="escalated")
            session.add(meta)
            
        audit_msg = TicketMessage(
            message_id=str(uuid.uuid4()),
            ticket_id=ticket_id,
            role=RoleEnum.system,
            content=f"[SYSTEM ESCALATION]: Ticket escalated to human support lead. Reason: {reason}"
        )
        session.add(audit_msg)
        session.commit()
        return {
            "success": True,
            "ticket_id": ticket_id,
            "status": "escalated",
            "reason": reason
        }
    except Exception as e:
        session.rollback()
        return {"error": str(e)}
    finally:
        session.close()
