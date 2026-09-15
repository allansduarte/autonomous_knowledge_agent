import sys
import os
from dotenv import load_dotenv

# Ensure root directory is on PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

from utils import chat_interface
from agentic.workflow import orchestrator

def main():
    print("=" * 60)
    print("      Uda-hub CultPass Agentic Customer Support Chat      ")
    print("=" * 60)
    ticket_id = input("Enter Ticket/Thread ID (default 'ticket_a4ab87'): ").strip()
    if not ticket_id:
        ticket_id = "ticket_a4ab87"
    
    print(f"\nStarting support session for Thread ID: {ticket_id}")
    print("Type your question below (or type 'exit' or 'quit' to end):\n")
    chat_interface(orchestrator, ticket_id)

if __name__ == "__main__":
    main()
