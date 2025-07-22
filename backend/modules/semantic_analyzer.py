# modules/semantic_analyzer.py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple
import logging

class AdvancedSemanticAnalyzer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize semantic analyzer with sentence transformers."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sentence_model = SentenceTransformer(model_name)
        self.sentence_model.to(self.device)
        
        # Load BERT for contextual analysis
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
        self.bert_model.to(self.device)
        
        logging.info(f"Semantic analyzer initialized on {self.device}")
    
    def extract_semantic_features(self, text: str) -> torch.Tensor:
        """Extract rich semantic features using BERT."""
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Use [CLS] token embedding for sentence-level representation
            sentence_embedding = outputs.last_hidden_state[:, 0, :]
        
        return sentence_embedding
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        embeddings = self.sentence_model.encode([text1, text2])
        similarity = F.cosine_similarity(
            torch.tensor(embeddings[0]).unsqueeze(0),
            torch.tensor(embeddings[1]).unsqueeze(0)
        ).item()
        
        return max(0.0, similarity)  # Ensure non-negative similarity
    
    def analyze_claim_evidence_alignment(self, claim: str, evidence_list: List[str]) -> Dict[str, float]:
        """Analyze how well evidence supports a claim."""
        claim_embedding = self.sentence_model.encode([claim])
        evidence_embeddings = self.sentence_model.encode(evidence_list)
        
        similarities = []
        for evidence_emb in evidence_embeddings:
            sim = F.cosine_similarity(
                torch.tensor(claim_embedding[0]).unsqueeze(0),
                torch.tensor(evidence_emb).unsqueeze(0)
            ).item()
            similarities.append(max(0.0, sim))
        
        return {
            'individual_similarities': similarities,
            'mean_similarity': np.mean(similarities),
            'max_similarity': np.max(similarities),
            'alignment_strength': self._calculate_alignment_strength(similarities)
        }
    
    def _calculate_alignment_strength(self, similarities: List[float]) -> str:
        """Determine alignment strength based on similarity scores."""
        mean_sim = np.mean(similarities)
        if mean_sim >= 0.8:
            return "very_strong"
        elif mean_sim >= 0.6:
            return "strong"
        elif mean_sim >= 0.4:
            return "moderate"
        elif mean_sim >= 0.2:
            return "weak"
        else:
            return "very_weak"
