import math
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


class TFIDFIndex:
    """Simple TF-IDF implementation using only numpy."""
    
    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self.idf_scores: np.ndarray = np.array([])
        self.tf_matrix: np.ndarray = np.array([])
        self.doc_count = 0
    
    def tokenize(self, text: str) -> List[str]:
        """Basic tokenization: lowercase, alphanumeric + common chars, min length 2."""
        import re
        # Include hyphens and underscores, split on whitespace and punctuation
        tokens = re.findall(r'[a-zA-Z0-9_-]{2,}', text.lower())
        return tokens
    
    def build_index(self, documents: List[str]) -> None:
        """Build TF-IDF index from documents."""
        self.doc_count = len(documents)
        if self.doc_count == 0:
            return
        
        # Build vocabulary
        all_tokens = []
        doc_tokens = []
        for doc in documents:
            tokens = self.tokenize(doc)
            doc_tokens.append(tokens)
            all_tokens.extend(tokens)
        
        # Create vocabulary mapping
        vocab_set = set(all_tokens)
        self.vocabulary = {token: i for i, token in enumerate(sorted(vocab_set))}
        vocab_size = len(self.vocabulary)
        
        # Build TF matrix
        self.tf_matrix = np.zeros((self.doc_count, vocab_size), dtype=np.float32)
        doc_frequencies = np.zeros(vocab_size, dtype=int)
        
        for doc_idx, tokens in enumerate(doc_tokens):
            token_counts = Counter(tokens)
            doc_length = len(tokens)
            
            for token, count in token_counts.items():
                if token in self.vocabulary:
                    token_idx = self.vocabulary[token]
                    # Log-normalized TF
                    self.tf_matrix[doc_idx, token_idx] = math.log(1 + count)
                    doc_frequencies[token_idx] += 1
        
        # Compute IDF scores
        self.idf_scores = np.zeros(vocab_size, dtype=np.float32)
        for i in range(vocab_size):
            if doc_frequencies[i] > 0:
                # Smoothed IDF
                self.idf_scores[i] = math.log(self.doc_count / (1 + doc_frequencies[i]))
    
    def query_vector(self, query: str) -> np.ndarray:
        """Convert query to TF-IDF vector."""
        if len(self.vocabulary) == 0:
            return np.array([])
        
        tokens = self.tokenize(query)
        token_counts = Counter(tokens)
        # query_length = len(tokens)
        
        query_vec = np.zeros(len(self.vocabulary), dtype=np.float32)
        for token, count in token_counts.items():
            if token in self.vocabulary:
                token_idx = self.vocabulary[token]
                # Log-normalized TF * IDF
                tf = math.log(1 + count)
                query_vec[token_idx] = tf * self.idf_scores[token_idx]
        
        return query_vec
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search documents using TF-IDF cosine similarity."""
        if self.doc_count == 0:
            print(f"TF-IDF: No documents indexed")
            return []
        
        query_tokens = self.tokenize(query)
        print(f"TF-IDF: Query tokens: {query_tokens}")
        
        query_vec = self.query_vector(query)
        if query_vec.size == 0:
            print(f"TF-IDF: Empty query vector")
            return []
        
        print(f"TF-IDF: Query vector non-zero elements: {np.count_nonzero(query_vec)}")
        
        # Cosine similarity
        doc_norms = np.linalg.norm(self.tf_matrix, axis=1)
        query_norm = np.linalg.norm(query_vec)
        
        if query_norm == 0:
            print(f"TF-IDF: Zero query norm")
            return []
        
        similarities = np.zeros(self.doc_count)
        for i in range(self.doc_count):
            if doc_norms[i] > 0:
                similarities[i] = np.dot(self.tf_matrix[i], query_vec) / (doc_norms[i] * query_norm)
        
        print(f"TF-IDF: Max similarity: {np.max(similarities):.4f}")
        
        # Get top-k results with lower threshold
        top_indices = np.argsort(-similarities)[:top_k]
        results = [(int(idx), float(similarities[idx])) for idx in top_indices if similarities[idx] > 0.01]  # Lower threshold
        print(f"TF-IDF: Returning {len(results)} results")
        return results


def hybrid_search(
    semantic_results: List[Dict], 
    tfidf_results: List[Tuple[int, float]], 
    chunk_texts: List[str],
    query: str,
    semantic_weight: float = 0.65,
    tfidf_weight: float = 0.35,
    threshold: float = 0.5
) -> List[Dict]:
    """Combine semantic and TF-IDF results with threshold filtering."""
    
    # Create mapping from chunk index to semantic score
    semantic_scores = {}
    for result in semantic_results:
        chunk_idx = result.get("metadata", {}).get("chunk_index", -1)
        if chunk_idx >= 0:
            semantic_scores[chunk_idx] = result["score"]
    
    # Create mapping from chunk index to TF-IDF score
    tfidf_scores = {}
    for idx, score in tfidf_results:
        tfidf_scores[idx] = score
    
    # Combine scores
    all_indices = set(semantic_scores.keys()) | set(tfidf_scores.keys())
    fused_results = []
    
    for chunk_idx in all_indices:
        semantic_score = semantic_scores.get(chunk_idx, 0.0)
        tfidf_score = tfidf_scores.get(chunk_idx, 0.0)
        
        # Normalize scores to [0, 1] range
        semantic_norm = max(0, min(1, semantic_score))
        tfidf_norm = max(0, min(1, tfidf_score))
        
        # Fused score
        fused_score = semantic_weight * semantic_norm + tfidf_weight * tfidf_norm
        
        # Boost for exact phrase matches
        if chunk_idx < len(chunk_texts):
            chunk_text = chunk_texts[chunk_idx].lower()
            query_lower = query.lower()
            if query_lower in chunk_text:
                fused_score *= 1.2  # 20% boost
        
        # Apply threshold
        if fused_score >= threshold:
            # Find original result to preserve metadata
            original_result = None
            for result in semantic_results:
                if result.get("metadata", {}).get("chunk_index") == chunk_idx:
                    original_result = result.copy()
                    break
            
            if original_result:
                original_result["score"] = fused_score
                original_result["semantic_score"] = semantic_norm
                original_result["tfidf_score"] = tfidf_norm
                fused_results.append(original_result)
    
    # Sort by fused score
    fused_results.sort(key=lambda x: x["score"], reverse=True)
    return fused_results
