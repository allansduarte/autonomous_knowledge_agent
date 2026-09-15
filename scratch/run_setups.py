import sys
import os
import json
import uuid
import random
from datetime import datetime, timedelta
from sqlalchemy import create_engine

from utils import reset_db, get_session
from data.models import cultpass, udahub

def setup_external_db():
    print("--- Setting up External DB (cultpass.db) ---")
    cultpass_db = "data/external/cultpass.db"
    reset_db(cultpass_db)
    engine = create_engine(f"sqlite:///{cultpass_db}", echo=False)
    cultpass.Base.metadata.create_all(engine)

    # 1. Experiences
    experience_data = []
    with open('data/external/cultpass_experiences.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                experience_data.append(json.loads(line))

    with get_session(engine) as session:
        experiences = []
        for idx, experience in enumerate(experience_data):
            exp = cultpass.Experience(
                experience_id=str(uuid.uuid4())[:6],
                title=experience["title"],
                description=experience["description"],
                location=experience["location"],
                when=datetime.now() + timedelta(days=idx+1),
                slots_available=random.randint(1, 30),
                is_premium=(idx % 2 == 0)
            )
            experiences.append(exp)
        session.add_all(experiences)

    # 2. Users & Subscriptions
    cultpass_users = []
    with open('data/external/cultpass_users.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                cultpass_users.append(json.loads(line))

    with get_session(engine) as session:
        db_users = []
        subscriptions = []
        for user_info in cultpass_users:
            user = cultpass.User(
                user_id=user_info["id"],
                full_name=user_info["name"],
                email=user_info["email"],
                is_blocked=user_info["is_blocked"],
                created_at=datetime.now()
            )
            db_users.append(user)

            subscription = cultpass.Subscription(
                subscription_id=str(uuid.uuid4())[:6],
                user_id=user_info["id"],
                status=random.choice(["active", "cancelled"]),
                tier=random.choice(["basic", "premium"]),
                monthly_quota=random.randint(2, 10),
                started_at=datetime.now()
            )
            subscriptions.append(subscription)

        session.add_all(db_users)
        session.add_all(subscriptions)

    # 3. Reservations for users
    with get_session(engine) as session:
        experience_ids = [exp.experience_id for exp in session.query(cultpass.Experience).all()]
        reservations = []
        for u in cultpass_users:
            for _ in range(random.randint(1, 3)):
                res = cultpass.Reservation(
                    reservation_id=str(uuid.uuid4())[:6],
                    user_id=u["id"],
                    experience_id=random.choice(experience_ids),
                    status=random.choice(["reserved", "completed", "cancelled"]),
                )
                reservations.append(res)
        session.add_all(reservations)

    print("✅ External DB Setup Complete.")

def setup_core_db():
    print("--- Setting up Core DB (udahub.db) ---")
    udahub_db = "data/core/udahub.db"
    reset_db(udahub_db)
    engine = create_engine(f"sqlite:///{udahub_db}", echo=False)
    udahub.Base.metadata.create_all(bind=engine)

    account_id = "cultpass"
    account_name = "CultPass Card"

    with get_session(engine) as session:
        account = udahub.Account(
            account_id=account_id,
            account_name=account_name,
        )
        session.add(account)

    # Load Knowledge Base
    cultpass_articles = []
    with open('data/external/cultpass_articles.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                cultpass_articles.append(json.loads(line))

    print(f"Loaded {len(cultpass_articles)} articles.")
    assert len(cultpass_articles) >= 14, "At least 14 articles required!"

    with get_session(engine) as session:
        kb = []
        for article in cultpass_articles:
            knowledge = udahub.Knowledge(
                article_id=str(uuid.uuid4()),
                account_id=account_id,
                title=article["title"],
                content=article["content"],
                tags=article["tags"]
            )
            kb.append(knowledge)
        session.add_all(kb)

    # Setup default tickets for cultpass users
    cultpass_users = []
    with open('data/external/cultpass_users.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                cultpass_users.append(json.loads(line))

    with get_session(engine) as session:
        for user_info in cultpass_users:
            user = udahub.User(
                user_id=str(uuid.uuid4()),
                account_id=account_id,
                external_user_id=user_info["id"],
                user_name=user_info["name"],
            )
            session.add(user)
            session.flush()

            ticket = udahub.Ticket(
                ticket_id=f"ticket_{user_info['id']}",
                account_id=account_id,
                user_id=user.user_id,
                channel="chat",
            )
            metadata = udahub.TicketMetadata(
                ticket_id=ticket.ticket_id,
                status="open",
                main_issue_type="general",
                tags="account, support",
            )
            first_message = udahub.TicketMessage(
                message_id=str(uuid.uuid4()),
                ticket_id=ticket.ticket_id,
                role="user",
                content=f"Hello, I am {user_info['name']}. I need assistance with my Cultpass account.",
            )
            session.add_all([ticket, metadata, first_message])

    print("✅ Core DB Setup Complete.")

if __name__ == "__main__":
    setup_external_db()
    setup_core_db()
