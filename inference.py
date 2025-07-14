#!/usr/bin/env python3
"""
DSRU Inference Script

Required files:
- model.py: Contains ScaledVectorReasoningEngine class
- vocabulary_helpers.py: Contains encode_texts, compute_similarities
- inference_questions.py: Contains TEST_CASES
- model.pt: Trained model checkpoint

Usage:
python inference.py --batch 32
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from sentence_transformers import SentenceTransformer
import time
from typing import List, Dict, Tuple, Optional
import numpy as np
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor

# Import model and helpers
from model import ScaledVectorReasoningEngine
from vocabulary_helpers import (
    encode_texts, 
    compute_similarities
)

# Import test cases
from inference_questions import TEST_CASES

# Model configuration
model_config = {
    'input_dim': 1024,
    'hidden_dim': 8192,
    'n_layers': 16
}


def cpu_decode_single_result(best_similarities_tensor: torch.Tensor, best_indices_tensor: torch.Tensor, vocabulary: List[str]) -> Tuple[str, float]:
    """
    Decode single result on CPU in separate thread.
    Does the CPU sync and label lookup.
    """
    best_idx = best_indices_tensor.item()  # CPU sync happens here
    best_sim = best_similarities_tensor.item()  # CPU sync happens here
    best_label = vocabulary[best_idx]  # Label lookup
    return best_label, best_sim


def load_model(checkpoint_path: str, device: str = 'cuda') -> ScaledVectorReasoningEngine:
    """Load the trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    model = ScaledVectorReasoningEngine(
        vector_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['n_layers']
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded. Trained for {checkpoint.get('total_batches_trained', 0):,} batches")
    return model


def encode_vocabulary_string(vocabulary: List[str], encoder: SentenceTransformer, device: str) -> torch.Tensor:
    """
    Encode vocabulary in the same format as training: "vocab1 | vocab2 | vocab3 | vocabn"
    """
    vocab_string = " | ".join(vocabulary)
    
    # Encode the vocabulary string
    vocab_embedding = encode_texts([vocab_string], encoder, device)
    return vocab_embedding


def compute_batch_similarities(query_vectors: torch.Tensor, candidate_vectors: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarities between multiple query vectors and candidate vectors.
    
    Args:
        query_vectors: Shape [batch_size, embedding_dim]
        candidate_vectors: Shape [n_candidates, embedding_dim]
        
    Returns:
        Similarities tensor of shape [batch_size, n_candidates]
    """
    
    # Normalize vectors if not already normalized
    norm_start = time.perf_counter()
    query_norm = F.normalize(query_vectors, p=2, dim=1)
    candidate_norm = F.normalize(candidate_vectors, p=2, dim=1)
    norm_time = time.perf_counter() - norm_start
    
    # Compute cosine similarity via matrix multiplication
    # query_norm: [batch_size, embedding_dim]
    # candidate_norm.T: [embedding_dim, n_candidates]
    # Result: [batch_size, n_candidates]
    matmul_start = time.perf_counter()
    similarities = torch.mm(query_norm, candidate_norm.t())
    matmul_time = time.perf_counter() - matmul_start
    return similarities


def predict_single_with_full_batching(
    task: str,
    data: str,
    vocabulary: List[str],
    model: ScaledVectorReasoningEngine,
    encoder: SentenceTransformer,
    device: str = 'cuda',
) -> Tuple[str, float, Dict[str, float]]:
    """
    Predict label for a single example with true batched encoding of all inputs.
    Measures FULL roundtrip time including CPU transfer and label decoding.
    
    Returns:
        (predicted_label, similarity, timing_dict)
    """
    timing = {}
    
    # Time the full encoding phase - batch all three inputs together
    encode_start = time.perf_counter()
    
    # Create vocabulary string
    vocab_string = " | ".join(vocabulary)
    
    # Batch encode ALL inputs together: task + data + vocab_string + individual vocab items
    all_texts = [task, data, vocab_string] + vocabulary
    all_embeddings = encode_texts(all_texts, encoder, device)
    
    # Split the batched results
    task_embedding = all_embeddings[0:1]  # Shape: [1, embedding_dim]
    data_embedding = all_embeddings[1:2]  # Shape: [1, embedding_dim]
    vocab_string_embedding = all_embeddings[2:3]  # Shape: [1, embedding_dim]
    vocab_embeddings = all_embeddings[3:]  # Shape: [vocab_size, embedding_dim]
    
    encode_time = time.perf_counter() - encode_start
    timing['encoding'] = encode_time
    
    # Time model inference
    inference_start = time.perf_counter()
    with torch.no_grad():
        predicted_vector = model(task_embedding, data_embedding, vocab_string_embedding)
        if predicted_vector.dim() > 1:
            predicted_vector = predicted_vector.squeeze()
    inference_time = time.perf_counter() - inference_start
    timing['model_inference'] = inference_time
    
    # Time vocabulary matching (GPU computation) - use exact same pattern as batch
    matching_start = time.perf_counter()
    
    # Use exact same pattern as batch version - no extra unsqueeze/squeeze operations
    predicted_vector_batched = predicted_vector.unsqueeze(0)  # [1, vector_dim]
    all_similarities = compute_batch_similarities(predicted_vector_batched, vocab_embeddings)
    # Result: [1, vocab_size]
    
    # Use same max operation as batch version - keep everything on GPU
    best_similarities, best_indices = all_similarities.max(dim=1)
    
    matching_time = time.perf_counter() - matching_start
    timing['vocab_matching'] = matching_time
    
    # Time the full async processing latency - from thread creation to result
    async_start = time.perf_counter()
    
    # Spin off CPU work in separate thread
    with ThreadPoolExecutor(max_workers=1) as executor:
        thread_submit_time = time.perf_counter()
        cpu_future = executor.submit(cpu_decode_single_result, best_similarities, best_indices, vocabulary)
        thread_creation_time = time.perf_counter() - thread_submit_time
        
        # GPU can immediately start next batch here if needed
        # For single examples, we just wait for the result
        
        wait_start = time.perf_counter()
        best_label, best_sim_cpu = cpu_future.result()  # Wait for CPU thread to complete
        total_async_time = time.perf_counter() - async_start
        wait_time = time.perf_counter() - wait_start
        
        timing['thread_creation'] = thread_creation_time
        timing['wait_time'] = wait_time
        timing['total_async_latency'] = total_async_time
        timing['cpu_transfer'] = total_async_time  # Keep this for compatibility
    
    # Total time includes everything
    timing['total'] = encode_time + inference_time + matching_time + total_async_time
    timing['per_example'] = timing['total']  # For single examples, same as total
    
    return best_label, best_sim_cpu, timing


def predict_batch_with_timing(
    task: str,
    data_batch: List[str],
    vocabulary: List[str],
    vocab_id: int,
    model: ScaledVectorReasoningEngine,
    encoder: SentenceTransformer,
    device: str = 'cuda',
    vocab_embeddings_cache: Dict[str, torch.Tensor] = None,
    task_embedding_cache: Dict[str, torch.Tensor] = None,
    async_transfer: bool = True
) -> Tuple[List[str], List[float], Dict[str, float], Optional[Tuple]]:
    """
    Predict labels for a batch of data with detailed timing.
    OPTIMIZED: Uses batched similarity computation for vocabulary matching.
    OPTIMIZED: Returns vocab_id with indices for async processing.
    
    Args:
        task: Task description
        data_batch: List of data strings to classify
        vocabulary: List of possible labels
        vocab_id: ID of the vocabulary for async processing
        model: The trained model
        encoder: Sentence encoder
        device: Device to run on
        vocab_embeddings_cache: Cache for vocabulary embeddings
        task_embedding_cache: Cache for task embeddings
        async_transfer: If True, return indices for async processing
    
    Returns:
        If async_transfer=False: (predicted_labels, similarities, timing_dict, None)
        If async_transfer=True: (None, None, timing_dict, (vocab_id, indices, similarities))
    """
    timing = {}
    batch_size = len(data_batch)
    
    # Setup phase (not counted in inference timing)
    setup_start = time.perf_counter()
    
    # Get or compute task embedding
    if task_embedding_cache is not None and task in task_embedding_cache:
        task_embedding = task_embedding_cache[task]
    else:
        task_embedding = encode_texts([task], encoder, device)
        if task_embedding_cache is not None:
            task_embedding_cache[task] = task_embedding
    
    # Replicate task embedding for batch
    task_embeddings = task_embedding.expand(batch_size, -1)
    
    # Get or compute vocabulary string embedding
    vocab_string = " | ".join(vocabulary)
    vocab_string_key = f"vocab_string:{vocab_string}"
    if vocab_embeddings_cache is not None and vocab_string_key in vocab_embeddings_cache:
        vocab_embedding = vocab_embeddings_cache[vocab_string_key]
    else:
        vocab_embedding = encode_vocabulary_string(vocabulary, encoder, device)
        if vocab_embeddings_cache is not None:
            vocab_embeddings_cache[vocab_string_key] = vocab_embedding
    
    # Replicate vocab embedding for batch
    vocab_embeddings_input = vocab_embedding.expand(batch_size, -1)
    
    # Get or compute individual vocabulary embeddings
    vocab_key = f"vocab_items:{vocab_string}"
    if vocab_embeddings_cache is not None and vocab_key in vocab_embeddings_cache:
        vocab_embeddings = vocab_embeddings_cache[vocab_key]
    else:
        vocab_embeddings = encode_texts(vocabulary, encoder, device)
        if vocab_embeddings_cache is not None:
            vocab_embeddings_cache[vocab_key] = vocab_embeddings
    
    setup_time = time.perf_counter() - setup_start
    timing['setup'] = setup_time
    
    # START OF ACTUAL INFERENCE TIMING
    # Time encoding phase (only data encoding counts)
    encode_start = time.perf_counter()
    data_embeddings = encode_texts(data_batch, encoder, device)
    encode_time = time.perf_counter() - encode_start
    timing['encoding'] = encode_time
    
    # Time model inference (batch)
    inference_start = time.perf_counter()
    with torch.no_grad():
        predicted_vectors = model(task_embeddings, data_embeddings, vocab_embeddings_input)
    inference_time = time.perf_counter() - inference_start
    timing['model_inference'] = inference_time
    
    # Time vocabulary matching - OPTIMIZED with batched computation
    matching_start = time.perf_counter()
    
    # Compute all similarities at once using batched operation
    # predicted_vectors: [batch_size, vector_dim]
    # vocab_embeddings: [vocab_size, vector_dim]
    all_similarities = compute_batch_similarities(predicted_vectors, vocab_embeddings)
    # all_similarities: [batch_size, vocab_size]
    
    # Get best matches for all examples at once
    best_similarities, best_indices = all_similarities.max(dim=1)
    
    matching_time = time.perf_counter() - matching_start
    timing['vocab_matching'] = matching_time
        
    # Total inference time (excluding setup)
    timing['total'] = encode_time + inference_time + matching_time
    timing['per_example'] = timing['total'] / batch_size
    
    if async_transfer:        
        # Return indices and similarities on GPU with vocab_id
        return None, None, timing, (vocab_id, best_indices, best_similarities)
    else:
        # Synchronous processing - transfer to CPU and decode
        indices_cpu = best_indices.cpu().numpy()
        similarities_cpu = best_similarities.cpu().numpy()
        
        # Convert to lists (AFTER timing ends - this is just output formatting)
        predicted_labels = [vocabulary[idx] for idx in indices_cpu]
        similarities_list = similarities_cpu.tolist()
        
        return predicted_labels, similarities_list, timing, None


def process_async_results(
    async_results: Tuple[int, torch.Tensor, torch.Tensor],
    vocabularies: Dict[int, List[str]]
) -> Tuple[List[str], List[float]]:
    """
    Process async results using the correct vocabulary.
    
    Args:
        async_results: (vocab_id, indices_tensor, similarities_tensor)
        vocabularies: Dictionary mapping vocab_id to vocabulary list
    
    Returns:
        (predicted_labels, similarities)
    """
    vocab_id, indices, similarities = async_results
    
    # Transfer to CPU
    indices_cpu = indices.cpu().numpy()
    similarities_cpu = similarities.cpu().numpy()
    
    # Look up labels using the correct vocabulary
    vocabulary = vocabularies[vocab_id]
    predicted_labels = [vocabulary[idx] for idx in indices_cpu]
    similarities_list = similarities_cpu.tolist()
    
    return predicted_labels, similarities_list


def analyze_prediction_error(
    task: str,
    data: str,
    vocabulary: List[str],
    predicted_label: str,
    correct_label: str,
    model: ScaledVectorReasoningEngine,
    encoder: SentenceTransformer,
    device: str = 'cuda'
):
    """Analyze why a prediction might be wrong."""
    print("\n" + "="*60)
    print("PREDICTION ERROR ANALYSIS")
    print("="*60)
    print(f"Task: {task[:80]}...")
    print(f"Data: {data[:80]}...")
    print(f"Predicted: {predicted_label}, Expected: {correct_label}")
    
    # Get the predicted vector
    task_embedding = encode_texts([task], encoder, device)
    data_embedding = encode_texts([data], encoder, device)
    vocab_embedding = encode_vocabulary_string(vocabulary, encoder, device)
    
    with torch.no_grad():
        predicted_vector = model(task_embedding, data_embedding, vocab_embedding)
        if predicted_vector.dim() > 1:
            predicted_vector = predicted_vector.squeeze()
    
    # Check similarities to all vocabulary items
    vocab_embeddings = encode_texts(vocabulary, encoder, device)
    similarities = compute_similarities(predicted_vector, vocab_embeddings)
    
    print("\nSimilarities to all labels:")
    for label, sim in zip(vocabulary, similarities):
        marker = "✓" if label == correct_label else ""
        print(f"  {label}: {sim.item():.4f} {marker}")


def compute_timing_statistics(timings: List[float]) -> Dict[str, float]:
    """Compute timing statistics in milliseconds."""
    timings_ms = [t * 1000 for t in timings]  # Convert to milliseconds
    return {
        'min': np.min(timings_ms),
        'max': np.max(timings_ms),
        'mean': np.mean(timings_ms),
        'median': np.median(timings_ms),
        'std': np.std(timings_ms)
    }


def run_classification_tests(batch_size: int = 1):
    """
    Run classification tests with detailed scorecard and timing analysis.
    
    Args:
        batch_size: Number of examples to process in each batch (1 = no batching)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = "model.pt"
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        return
    
    # Load model and encoder
    print("="*70)
    print("THREE-HEADED VECTOR REASONING ENGINE - INFERENCE WITH TIMING")
    print("="*70)
    print(f"Batch size: {batch_size}")
    model = load_model(checkpoint_path, device)
    encoder = SentenceTransformer('BAAI/bge-large-en-v1.5', device=device)
    print(f"Encoder loaded on {device}")
    
    # Create global vocabulary store
    print("\nCreating global vocabulary store...")
    test_cases = TEST_CASES
    vocabularies = {}
    vocab_id_counter = 0
    
    for test_case in test_cases:
        vocabulary = test_case["vocabulary"]
        vocabularies[vocab_id_counter] = vocabulary
        test_case["vocab_id"] = vocab_id_counter
        vocab_id_counter += 1
    
    print(f"Created {len(vocabularies)} vocabulary mappings")
    
    # Warm up the model and encoder (important for accurate timing)
    print("\nWarming up model and encoder...")
    warmup_task = "Classify this"
    warmup_data = ["This is a warmup"] * batch_size
    warmup_vocab = ["A", "B"]
    warmup_vocab_id = -1  # Special ID for warmup
    
    for _ in range(5):
        predict_batch_with_timing(
            warmup_task, warmup_data, warmup_vocab, warmup_vocab_id,
            model, encoder, device,
            async_transfer=False
        )
    print("Warmup complete.")
    
    # Run tests and collect detailed results
    print("\n" + "="*70)
    print("RUNNING INFERENCE WITH TIMING")
    print("="*70)
    
    all_results = []
    task_summaries = []
    all_timings = {
        'encoding': [],
        'model_inference': [],
        'vocab_matching': [],
        'cpu_transfer': [],
        'total': [],
        'per_example': []
    }
    
    # Create caches for embeddings
    vocab_embeddings_cache = {}
    task_embedding_cache = {}
    
    for test_idx, test_case in enumerate(test_cases):
        print(f"\n{'='*50}")
        print(f"TASK: {test_case['name']}")
        print(f"{'='*50}")
        
        task = test_case["task"]
        vocabulary = test_case["vocabulary"]
        vocab_id = test_case["vocab_id"]
        print(f"Task: {task[:80]}...")
        print(f"Labels: {vocabulary}")
        
        task_correct = 0
        task_timings = []
        task_per_example_timings = []
        
        # Process examples in batches
        examples = test_case["examples"]
        num_examples = len(examples)
        
        # For async processing, store pending results
        pending_results = []
        
        for batch_start in range(0, num_examples, batch_size):
            batch_end = min(batch_start + batch_size, num_examples)
            batch_examples = examples[batch_start:batch_end]
            batch_data = [ex[0] for ex in batch_examples]
            batch_expected = [ex[1] for ex in batch_examples]
            
            # Initialize async_results to None for all cases
            async_results = None
            
            if len(batch_data) == 1:
                # Single example - use fully batched encoding with full roundtrip timing
                data, expected_label = batch_data[0], batch_expected[0]
                
                # Use the new fully batched single prediction function
                pred_label, pred_sim, timing = predict_single_with_full_batching(
                    task, data, vocabulary,
                    model, encoder, device,
                )
                
                predicted_labels = [pred_label]
                similarities_list = [pred_sim]
            else:
                # Batch prediction
                predicted_labels, similarities_list, timing, async_results = predict_batch_with_timing(
                    task, batch_data, vocabulary, vocab_id,
                    model, encoder, device,
                    vocab_embeddings_cache=vocab_embeddings_cache,
                    task_embedding_cache=task_embedding_cache,
                    async_transfer=True
                )
                
                if async_results is not None:
                    # Store for later processing
                    pending_results.append((async_results, batch_data, batch_expected, timing))
                    
                    # Still need to store timing data!
                    timing_keys = ['encoding', 'model_inference', 'vocab_matching', 'total']
                    for key in timing_keys:
                        if key in timing:
                            all_timings[key].append(timing[key])
                    all_timings['per_example'].extend([timing['per_example']] * len(batch_data))
                    task_timings.append(timing['total'])
                    task_per_example_timings.extend([timing['per_example']] * len(batch_data))
                    
                    continue  # Skip immediate result processing
            
            # Store timing data
            timing_keys = ['encoding', 'model_inference', 'vocab_matching', 'total']
            if 'cpu_transfer' in timing:  # Single example includes CPU transfer
                timing_keys.append('cpu_transfer')
            
            for key in timing_keys:
                if key in timing:
                    all_timings[key].append(timing[key])
            all_timings['per_example'].extend([timing['per_example']] * len(batch_data))
            task_timings.append(timing['total'])
            task_per_example_timings.extend([timing['per_example']] * len(batch_data))
            
            # Process results for each example in batch (sync path)
            if async_results is None:
                for idx, (data, expected_label, pred_label, pred_sim) in enumerate(
                    zip(batch_data, batch_expected, predicted_labels, similarities_list)
                ):
                    pred_dist = 1 - pred_sim
                    
                    # Calculate distance to correct answer (for error analysis)
                    task_emb = encode_texts([task], encoder, device)
                    data_emb = encode_texts([data], encoder, device)
                    vocab_emb = encode_vocabulary_string(vocabulary, encoder, device)
                    
                    with torch.no_grad():
                        pred_vector = model(task_emb, data_emb, vocab_emb).squeeze()
                    
                    # Get similarity to correct label
                    vocab_embeddings = encode_texts(vocabulary, encoder, device)
                    correct_idx = vocabulary.index(expected_label)
                    correct_sim = F.cosine_similarity(
                        pred_vector.unsqueeze(0), 
                        vocab_embeddings[correct_idx].unsqueeze(0)
                    ).item()
                    correct_dist = 1 - correct_sim
                    
                    # Calculate difference
                    difference = correct_dist - pred_dist
                    
                    # Check if correct
                    is_correct = pred_label == expected_label
                    if is_correct:
                        task_correct += 1
                        result_str = "✓"
                    else:
                        result_str = "✗"
                    
                    # Store result
                    all_results.append({
                        "task": test_case["name"],
                        "data": data,
                        "expected": expected_label,
                        "predicted": pred_label,
                        "correct_dist": correct_dist,
                        "pred_dist": pred_dist,
                        "difference": difference,
                        "is_correct": is_correct,
                        "inference_time_ms": timing['per_example'] * 1000
                    })
                    
                    # Print example result (only first 5)
                    example_num = batch_start + idx + 1
                    if example_num <= 5:
                        print(f"\n  Example {example_num}:")
                        print(f"  Data: \"{data[:60]}...\"")
                        print(f"  Expected: {expected_label}, Predicted: {pred_label}")
                        print(f"  Distances - Predicted: {pred_dist:.4f}, Correct: {correct_dist:.4f}")
                        print(f"  Difference: {difference:+.4f} {result_str}")
                        print(f"  Inference time: {timing['per_example']*1000:.2f}ms")
        
        # Process any pending async results for this task
        if pending_results:
            for async_results, batch_data, batch_expected, timing in pending_results:
                # Process results using correct vocabulary
                predicted_labels, similarities_list = process_async_results(
                    async_results, vocabularies
                )
                
                # Process results for each example
                for idx, (data, expected_label, pred_label, pred_sim) in enumerate(
                    zip(batch_data, batch_expected, predicted_labels, similarities_list)
                ):
                    pred_dist = 1 - pred_sim
                    
                    # Calculate distance to correct answer (for error analysis)
                    task_emb = encode_texts([task], encoder, device)
                    data_emb = encode_texts([data], encoder, device)
                    vocab_emb = encode_vocabulary_string(vocabulary, encoder, device)
                    
                    with torch.no_grad():
                        pred_vector = model(task_emb, data_emb, vocab_emb).squeeze()
                    
                    # Get similarity to correct label
                    vocab_embeddings = encode_texts(vocabulary, encoder, device)
                    correct_idx = vocabulary.index(expected_label)
                    correct_sim = F.cosine_similarity(
                        pred_vector.unsqueeze(0), 
                        vocab_embeddings[correct_idx].unsqueeze(0)
                    ).item()
                    correct_dist = 1 - correct_sim
                    
                    # Calculate difference
                    difference = correct_dist - pred_dist
                    
                    # Check if correct
                    is_correct = pred_label == expected_label
                    if is_correct:
                        task_correct += 1
                        result_str = "✓"
                    else:
                        result_str = "✗"
                    
                    # Store result
                    all_results.append({
                        "task": test_case["name"],
                        "data": data,
                        "expected": expected_label,
                        "predicted": pred_label,
                        "correct_dist": correct_dist,
                        "pred_dist": pred_dist,
                        "difference": difference,
                        "is_correct": is_correct,
                        "inference_time_ms": timing['per_example'] * 1000
                    })
        
        # Task summary with timing
        task_accuracy = task_correct / len(test_case["examples"]) * 100
        task_timing_stats = compute_timing_statistics(task_per_example_timings)
        task_summaries.append({
            "name": test_case["name"],
            "correct": task_correct,
            "total": len(test_case["examples"]),
            "accuracy": task_accuracy,
            "timing_stats": task_timing_stats
        })
        print(f"\n  Task Score: {task_correct}/{len(test_case['examples'])} ({task_accuracy:.1f}%)")
        print(f"  Task Timing per example (ms): min={task_timing_stats['min']:.2f}, "
              f"max={task_timing_stats['max']:.2f}, "
              f"mean={task_timing_stats['mean']:.2f}, "
              f"median={task_timing_stats['median']:.2f}")
    
    # Print detailed scorecard
    print("\n" + "="*100)
    print("FINAL SCORECARD - ALL QUESTIONS")
    print("="*100)
    print(f"{'Task':<30} {'Expected':<20} {'Predicted':<20} {'Result':<8} {'Difference':<12}")
    print("-" * 90)
    
    total_correct = 0
    all_differences = []
    
    for result in all_results:
        total_correct += result["is_correct"]
        all_differences.append(result["difference"])
        
        result_str = "✓" if result["is_correct"] else "✗"
        # Only show errors in final scorecard
        if not result["is_correct"]:
            print(f"{result['task']:<30} {result['expected']:<20} {result['predicted']:<20} "
                  f"{result_str:<8} {result['difference']:+12.4f}")
    
    print("-" * 90)
    
    # Summary statistics
    total_tests = len(all_results)
    overall_accuracy = total_correct / total_tests * 100
    
    # Separate statistics for incorrect only
    incorrect_differences = [r['difference'] for r in all_results if not r['is_correct']]
    avg_diff_when_incorrect = sum(incorrect_differences) / len(incorrect_differences) if incorrect_differences else 0
    
    print(f"{'TOTAL CORRECT':<30} {'':<20} {'':<20} {total_correct}/{total_tests}")
    print(f"{'OVERALL ACCURACY':<30} {'':<20} {'':<20} {overall_accuracy:.1f}%")
    if incorrect_differences:
        print(f"{'AVG DIFF (ERRORS ONLY)':<30} {'':<20} {'':<20} {'':<8} {avg_diff_when_incorrect:+12.4f}")
    print("="*100)
    
    # Timing statistics
    print("\n" + "="*70)
    print(f"TIMING STATISTICS (milliseconds) - Batch Size: {batch_size}")
    print("="*70)
    
    # Show per-batch timing
    print("\nPER-BATCH TIMING:")
    timing_components = ['encoding', 'model_inference', 'vocab_matching', 'total']
    if all_timings['cpu_transfer']:
        timing_components.insert(-1, 'cpu_transfer')
    
    for component in timing_components:
        if all_timings[component]:
            stats = compute_timing_statistics(all_timings[component])
            print(f"\n{component.upper()}:")
            print(f"  Min:    {stats['min']:7.2f} ms")
            print(f"  Max:    {stats['max']:7.2f} ms")
            print(f"  Mean:   {stats['mean']:7.2f} ms")
            print(f"  Median: {stats['median']:7.2f} ms")
            print(f"  Std:    {stats['std']:7.2f} ms")
    
    # Show per-example timing
    print("\n\nPER-EXAMPLE TIMING:")
    stats = compute_timing_statistics(all_timings['per_example'])
    print(f"  Min:    {stats['min']:7.2f} ms")
    print(f"  Max:    {stats['max']:7.2f} ms")
    print(f"  Mean:   {stats['mean']:7.2f} ms")
    print(f"  Median: {stats['median']:7.2f} ms")
    print(f"  Std:    {stats['std']:7.2f} ms")
    
    # Show breakdown percentages
    print("\nAverage Time Breakdown (per batch):")
    total_mean = np.mean(all_timings['total']) * 1000
    breakdown_components = ['encoding', 'model_inference', 'vocab_matching']
    if all_timings['cpu_transfer']:
        breakdown_components.append('cpu_transfer')
    
    for component in breakdown_components:
        if all_timings[component]:
            component_mean = np.mean(all_timings[component]) * 1000
            percentage = (component_mean / total_mean) * 100
            print(f"  {component}: {component_mean:.2f}ms ({percentage:.1f}%)")
    
    # Show batch processing info
    if batch_size > 1:
        print(f"\nBatch Processing:")
        per_example_time = np.mean(all_timings['per_example']) * 1000
        batch_total_time = np.mean(all_timings['total']) * 1000
        print(f"  Average batch size: {batch_size}")
        print(f"  Time per batch: {batch_total_time:.2f}ms")
        print(f"  Time per example in batch: {per_example_time:.2f}ms")
    
    # Task summary table
    print("\n" + "="*70)
    print("TASK SUMMARY WITH TIMING")
    print("="*70)
    print(f"{'Task':<35} {'Correct':<10} {'Total':<10} {'Accuracy':<10} {'Med Time (ms)':<15}")
    print("-" * 80)
    
    for summary in task_summaries:
        print(f"{summary['name']:<35} {summary['correct']:<10} {summary['total']:<10} "
              f"{summary['accuracy']:<10.1f}% {summary['timing_stats']['median']:<15.2f}")
    
    print("-" * 80)
    print(f"{'OVERALL':<35} {total_correct:<10} {total_tests:<10} {overall_accuracy:<10.1f}%")
    print("="*70)
    
    # Model resource summary
    print("\n" + "="*70)
    print("MODEL RESOURCE SUMMARY")
    print("="*70)
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate memory footprint
    param_memory_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    # Get actual GPU memory usage if on CUDA
    if device.startswith('cuda'):
        torch.cuda.synchronize()
        allocated_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        reserved_mb = torch.cuda.memory_reserved(device) / (1024 * 1024)
        
    print(f"Model Architecture:")
    print(f"  Input dimension: {model_config['input_dim']}")
    print(f"  Hidden dimension: {model_config['hidden_dim']}")
    print(f"  Number of layers: {model_config['n_layers']}")
    print(f"  Three input projections: {model_config['input_dim']} → 4096 each")
    print(f"\nParameter Count:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"\nMemory Footprint:")
    print(f"  Model parameters: {param_memory_mb:.2f} MB")
    if device.startswith('cuda'):
        print(f"  GPU memory allocated: {allocated_mb:.2f} MB")
        print(f"  GPU memory reserved: {reserved_mb:.2f} MB")
    print("="*70)
    
    # Find slowest and fastest examples
    print("\n" + "="*70)
    print("EXTREME TIMING EXAMPLES (per-example time)")
    print("="*70)
    
    sorted_by_time = sorted(all_results, key=lambda x: x['inference_time_ms'])
    
    print("\nFastest 3 inferences:")
    for i in range(min(3, len(sorted_by_time))):
        result = sorted_by_time[i]
        print(f"  {result['inference_time_ms']:.2f}ms - Task: {result['task'][:30]}, "
              f"Data: \"{result['data'][:40]}...\"")
    
    print("\nSlowest 3 inferences:")
    for i in range(max(0, len(sorted_by_time)-3), len(sorted_by_time)):
        result = sorted_by_time[i]
        print(f"  {result['inference_time_ms']:.2f}ms - Task: {result['task'][:30]}, "
              f"Data: \"{result['data'][:40]}...\"")
    
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference with timing measurements')
    parser.add_argument('--batch', type=int, default=1, 
                        help='Batch size for inference (default: 1, no batching)')
     
    args = parser.parse_args()
    run_classification_tests(batch_size=args.batch)
