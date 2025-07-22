# config.py - Shared component instances
"""
Global component instances for VerifAI enhanced system.
This prevents circular imports between app.py and routes.
"""

# Global component instances (initialized in app.py)
semantic_analyzer = None
content_extractor = None
evidence_retriever = None
credibility_assessor = None
evidence_aggregator = None
verdict_generator = None
