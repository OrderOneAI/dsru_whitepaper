#!/usr/bin/env python3
"""
Helper functions for vocabulary management and similarity calculations.
Clean, decomposable functions with clear inputs and outputs.
FIXED: compute_similarities now uses efficient matrix multiplication like compute_batch_similarities.
"""
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import time


def encode_texts(texts: List[str], encoder: SentenceTransformer, device: str = 'cuda') -> torch.Tensor:
    """
    Encode a list of texts into normalized embeddings.
    
    Args:
        texts: List of strings to encode
        encoder: SentenceTransformer model
        device: Device to put embeddings on
        
    Returns:
        Tensor of shape [len(texts), embedding_dim] with normalized embeddings
    """
    embeddings = encoder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_tensor=True
    )
    return embeddings.to(device)


def flatten_vocabulary(vocab_dict: Dict[str, List[str]]) -> Tuple[List[str], Dict[str, str]]:
    """
    Flatten a vocabulary dictionary into a list of terms and a mapping back to labels.
    
    Args:
        vocab_dict: Dictionary mapping labels to lists of terms
        
    Returns:
        all_terms: Flat list of all terms
        term_to_label: Dictionary mapping each term to its label
    """
    all_terms = []
    term_to_label = {}
    
    for label, terms in vocab_dict.items():
        for term in terms:
            all_terms.append(term)
            term_to_label[term] = label
            
    return all_terms, term_to_label


def compute_batch_similarities(query_vectors: torch.Tensor, candidate_vectors: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarities between multiple query vectors and candidate vectors.
    
    Args:
        query_vectors: Shape [batch_size, embedding_dim]
        candidate_vectors: Shape [n_candidates, embedding_dim]
        
    Returns:
        Similarities tensor of shape [batch_size, n_candidates]
    """
    print(f"[COMPUTE_BATCH_SIM] Input shapes: query={query_vectors.shape}, candidates={candidate_vectors.shape}")
    
    # Normalize vectors if not already normalized
    norm_start = time.perf_counter()
    query_norm = F.normalize(query_vectors, p=2, dim=1)
    candidate_norm = F.normalize(candidate_vectors, p=2, dim=1)
    norm_time = time.perf_counter() - norm_start
    print(f"[COMPUTE_BATCH_SIM] Normalization time: {norm_time*1000:.2f}ms")
    
    # Compute cosine similarity via matrix multiplication
    matmul_start = time.perf_counter()
    similarities = torch.mm(query_norm, candidate_norm.t())
    matmul_time = time.perf_counter() - matmul_start
    print(f"[COMPUTE_BATCH_SIM] Matrix multiplication time: {matmul_time*1000:.2f}ms")
    
    print(f"[COMPUTE_BATCH_SIM] Output shape: {similarities.shape}")
    return similarities


def compute_similarities(query_vector: torch.Tensor, candidate_vectors: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarities between a query vector and candidate vectors.
    OPTIMIZED: Now uses efficient matrix multiplication like compute_batch_similarities.
    
    Args:
        query_vector: Shape [embedding_dim] or [1, embedding_dim]
        candidate_vectors: Shape [n_candidates, embedding_dim]
        
    Returns:
        Similarities tensor of shape [n_candidates]
    """
    # Ensure query_vector is 1D, then add batch dimension
    if query_vector.dim() > 1:
        query_vector = query_vector.squeeze()
    
    # Use the same efficient matrix multiplication approach as compute_batch_similarities
    query_norm = F.normalize(query_vector.unsqueeze(0), p=2, dim=1)  # [1, embedding_dim]
    candidate_norm = F.normalize(candidate_vectors, p=2, dim=1)       # [n_candidates, embedding_dim]
    
    # Matrix multiplication - same as batched version
    similarities = torch.mm(query_norm, candidate_norm.t())  # [1, n_candidates]
    
    return similarities.squeeze(0)  # [n_candidates]


def find_top_k_terms(
    similarities: torch.Tensor,
    all_terms: List[str],
    term_to_label: Dict[str, str],
    k: int = 5
) -> List[Tuple[str, str, float]]:
    """
    Find the top-k terms based on similarities.
    
    Args:
        similarities: Tensor of similarities for each term
        all_terms: List of all terms (same order as similarities)
        term_to_label: Mapping from terms to labels
        k: Number of top results to return
        
    Returns:
        List of (term, label, similarity) tuples, sorted by similarity descending
    """
    # Get top-k
    k = min(k, len(all_terms))
    top_similarities, top_indices = torch.topk(similarities, k)
    
    results = []
    for idx, sim in zip(top_indices.cpu().numpy(), top_similarities.cpu().numpy()):
        term = all_terms[idx]
        label = term_to_label[term]
        results.append((term, label, float(sim)))
    
    return results


def get_similarities_for_label(
    query_vector: torch.Tensor,
    label: str,
    vocab_dict: Dict[str, List[str]],
    encoder: SentenceTransformer,
    device: str = 'cuda'
) -> List[Tuple[str, float]]:
    """
    Get similarities for all terms in a specific label.
    
    Args:
        query_vector: The query embedding
        label: The label to check
        vocab_dict: Vocabulary dictionary
        encoder: Sentence transformer
        device: Device to use
        
    Returns:
        List of (term, similarity) tuples for that label
    """
    terms = vocab_dict[label]
    term_embeddings = encode_texts(terms, encoder, device)
    similarities = compute_similarities(query_vector, term_embeddings)
    
    results = []
    for term, sim in zip(terms, similarities):
        results.append((term, sim.item()))
    
    return results


def debug_vocabulary_search(
    query_vector: torch.Tensor,
    vocab_dict: Dict[str, List[str]],
    encoder: SentenceTransformer,
    device: str = 'cuda',
    expected_label: str = None
):
    """
    Debug function to show all similarities and identify potential issues.
    
    Args:
        query_vector: The query embedding
        vocab_dict: Vocabulary dictionary
        encoder: Sentence transformer
        device: Device to use
        expected_label: If provided, highlight terms from this label
    """
    print("\n[DEBUG] Vocabulary Search Analysis")
    print("=" * 50)
    
    # Flatten vocabulary
    all_terms, term_to_label = flatten_vocabulary(vocab_dict)
    print(f"Total terms: {len(all_terms)}")
    print(f"Labels: {list(vocab_dict.keys())}")
    
    # Encode all terms
    all_embeddings = encode_texts(all_terms, encoder, device)
    print(f"Embeddings shape: {all_embeddings.shape}")
    
    # Compute all similarities
    similarities = compute_similarities(query_vector, all_embeddings)
    print(f"Similarities shape: {similarities.shape}")
    
    # Sort by similarity
    sorted_indices = torch.argsort(similarities, descending=True)
    
    print("\nAll terms sorted by similarity:")
    for i, idx in enumerate(sorted_indices):
        term = all_terms[idx]
        label = term_to_label[term]
        sim = similarities[idx].item()
        dist = 1 - sim
        
        marker = ""
        if expected_label and label == expected_label:
            marker = " ← EXPECTED"
        if i == 0:
            marker += " ← TOP PICK"
            
        print(f"{i+1:3d}. '{term}' [{label}]: sim={sim:.6f}, dist={dist:.6f}{marker}")
    
    # Analysis by label
    print("\nBest similarity per label:")
    for label in vocab_dict:
        label_terms = vocab_dict[label]
        label_indices = [i for i, term in enumerate(all_terms) if term_to_label[term] == label]
        label_sims = similarities[label_indices]
        best_sim = label_sims.max().item()
        best_idx = label_indices[label_sims.argmax()]
        best_term = all_terms[best_idx]
        print(f"  {label}: '{best_term}' (sim={best_sim:.6f}, dist={1-best_sim:.6f})")
