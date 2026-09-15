import os
from typing import Dict, Any, List, Union
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_core.tools import tool

# Determine project root dynamically
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CULTPASS_DB_PATH = os.path.join(PROJECT_ROOT, "data", "external", "cultpass.db")

def get_cultpass_engine():
    return create_engine(f"sqlite:///{CULTPASS_DB_PATH}", echo=False)

@tool
def get_user_profile(user_id_or_email: str) -> Dict[str, Any]:
    """Retrieves a CultPass user profile including subscription details and status by user ID or email."""
    if not user_id_or_email or not str(user_id_or_email).strip():
        return {"error": "User identifier cannot be empty."}
        
    user_id_or_email = str(user_id_or_email).strip()
    from data.models.cultpass import User
    
    try:
        engine = get_cultpass_engine()
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            user = session.query(User).filter(
                (User.user_id == user_id_or_email) | (User.email == user_id_or_email)
            ).first()
            if not user:
                return {"error": f"User '{user_id_or_email}' not found."}
            
            sub_info = None
            if user.subscription:
                sub_info = {
                    "subscription_id": user.subscription.subscription_id,
                    "status": user.subscription.status,
                    "tier": user.subscription.tier,
                    "monthly_quota": user.subscription.monthly_quota,
                    "started_at": str(user.subscription.started_at),
                }
            
            return {
                "user_id": user.user_id,
                "full_name": user.full_name,
                "email": user.email,
                "is_blocked": user.is_blocked,
                "subscription": sub_info,
            }
        finally:
            session.close()
    except Exception as e:
        return {"error": f"Database error: {str(e)}"}

@tool
def get_user_reservations(user_id: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """Retrieves all reservations for a given CultPass user ID.
    Validates user ID and distinguishes unknown accounts from accounts with zero reservations.
    """
    if not user_id or not str(user_id).strip():
        return {"error": "User identifier cannot be empty."}
        
    user_id = str(user_id).strip()
    from data.models.cultpass import User, Reservation
    
    try:
        engine = get_cultpass_engine()
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            # Check if account exists first
            user = session.query(User).filter(User.user_id == user_id).first()
            if not user:
                return {"error": f"User '{user_id}' not found."}
                
            reservations = session.query(Reservation).filter(Reservation.user_id == user_id).all()
            results = []
            for r in reservations:
                exp_title = r.experience.title if r.experience else "Unknown Experience"
                results.append({
                    "reservation_id": r.reservation_id,
                    "user_id": r.user_id,
                    "experience_id": r.experience_id,
                    "experience_title": exp_title,
                    "status": r.status,
                    "created_at": str(r.created_at),
                })
            return results
        finally:
            session.close()
    except Exception as e:
        return {"error": f"Database error: {str(e)}"}

@tool
def search_experiences(query: str = "") -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """Searches available CultPass experiences/events by title or location."""
    from data.models.cultpass import Experience
    try:
        engine = get_cultpass_engine()
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            q = session.query(Experience)
            if query and str(query).strip():
                clean_query = str(query).strip()
                q = q.filter(
                    (Experience.title.ilike(f"%{clean_query}%")) | 
                    (Experience.location.ilike(f"%{clean_query}%")) |
                    (Experience.description.ilike(f"%{clean_query}%"))
                )
            experiences = q.all()
            return [
                {
                    "experience_id": e.experience_id,
                    "title": e.title,
                    "description": e.description,
                    "location": e.location,
                    "when": str(e.when),
                    "slots_available": e.slots_available,
                    "is_premium": e.is_premium,
                }
                for e in experiences
            ]
        finally:
            session.close()
    except Exception as e:
        return {"error": f"Database error: {str(e)}"}
