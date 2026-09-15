import os
import re
from typing import Dict, Any, List
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_core.tools import tool

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UDAHUB_DB_PATH = os.path.join(PROJECT_ROOT, "data", "core", "udahub.db")

def get_udahub_engine():
    return create_engine(f"sqlite:///{UDAHUB_DB_PATH}", echo=False)

def compute_similarity(query: str, title: str, tags: str, content: str) -> float:
    """Computes a TF-IDF keyword & topic matching score normalized between 0.0 and 1.0."""
    query_terms = set(re.findall(r'\w+', query.lower()))
    if not query_terms:
        return 0.0
    
    title_terms = set(re.findall(r'\w+', title.lower()))
    tags_terms = set(re.findall(r'\w+', (tags or "").lower()))
    content_terms = re.findall(r'\w+', content.lower())
    
    raw_score = 0.0
    for term in query_terms:
        if term in title_terms:
            raw_score += 4.0
        if term in tags_terms:
            raw_score += 3.0
        count = content_terms.count(term)
        if count > 0:
            raw_score += 1.0 + min(count * 0.5, 3.0)
            
    # Normalize score relative to max potential score
    max_potential = len(query_terms) * 8.0
    normalized = min(raw_score / max_potential, 1.0) if max_potential > 0 else 0.0
    return round(normalized, 2)

@tool
def search_knowledge_base(query: str, top_k: int = 3, min_confidence: float = 0.25) -> Dict[str, Any]:
    """Performs RAG search over CultPass knowledge articles in Udahub DB.
    Includes confidence scoring and explicit escalation flags if no sufficiently confident match is found.
    """
    from data.models.udahub import Knowledge
    engine = get_udahub_engine()
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        articles = session.query(Knowledge).all()
        scored_articles = []
        
        for art in articles:
            conf = compute_similarity(query, art.title, art.tags, art.content)
            scored_articles.append({
                "article_id": art.article_id,
                "title": art.title,
                "tags": art.tags,
                "content": art.content,
                "confidence_score": conf
            })
            
        # Sort by confidence score descending
        scored_articles.sort(key=lambda x: x["confidence_score"], reverse=True)
        
        top_matches = [a for a in scored_articles if a["confidence_score"] >= min_confidence][:top_k]
        
        # Escalation logic: If top match is below confidence threshold, recommend escalation
        should_escalate = len(top_matches) == 0 or (scored_articles and scored_articles[0]["confidence_score"] < min_confidence)
        highest_confidence = scored_articles[0]["confidence_score"] if scored_articles else 0.0
        
        return {
            "query": query,
            "highest_confidence": highest_confidence,
            "should_escalate": should_escalate,
            "escalation_reason": "Low knowledge confidence score (no matching article found above threshold)." if should_escalate else None,
            "articles": top_matches if top_matches else scored_articles[:top_k]
        }
    finally:
        session.close()
