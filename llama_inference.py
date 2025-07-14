#!/usr/bin/env python3
"""
Benchmark Zephyr-7B on classification tasks.
Provides direct comparison with DSRU performance.
"""
import torch
import time
import json
import argparse
import re
from typing import List, Dict, Tuple
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

# Import test cases from your existing file
from inference_questions import TEST_CASES


def load_zephyr_model(device: str = 'cuda'):
    """Load Zephyr-7B model."""
    model_path = "HuggingFaceH4/zephyr-7b-beta"
    print(f"Loading Zephyr-7B from {model_path}...")
    print("This model does not require HuggingFace authentication!")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded successfully on {device}")
    return model, tokenizer


def create_classification_prompt(task: str, data: str, vocabulary: List[str]) -> str:
    """Create a prompt for Zephyr to perform classification."""
    vocab_str = ", ".join([f'"{v}"' for v in vocabulary])
    
    prompt = f"""<|system|>
You are a classification system. Output only the exact label.
<|user|>
Task: {task}
Text: "{data}"
Labels: {vocab_str}
<|assistant|>"""
    
    return prompt


def classify_with_zephyr(
    model,
    tokenizer,
    task: str,
    data: str,
    vocabulary: List[str],
    device: str = 'cuda'
) -> Tuple[str, float, Dict[str, float], str]:
    """
    Classify text using Zephyr-7B.
    
    Returns:
        (predicted_label, confidence, timing_dict, raw_response)
    """
    timing = {}
    
    # Create prompt
    prompt = create_classification_prompt(task, data, vocabulary)
    
    # Time tokenization
    tokenize_start = time.perf_counter()
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_length = inputs['input_ids'].shape[1]
    tokenize_time = time.perf_counter() - tokenize_start
    timing['tokenization'] = tokenize_time
    
    # Time generation
    generate_start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    generate_time = time.perf_counter() - generate_start
    timing['generation'] = generate_time
    
    # Time decoding
    decode_start = time.perf_counter()
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    decode_time = time.perf_counter() - decode_start
    timing['decoding'] = decode_time
    
    # Store raw response
    raw_response = response
    
    # Parse response
    predicted_label = None
    confidence = 0.0
    
    # Clean response
    response_clean = response.strip()
    for prefix in ['Label:', 'The label is:', 'Answer:']:
        if response_clean.startswith(prefix):
            response_clean = response_clean[len(prefix):].strip()
    
    response_clean = response_clean.strip('"\'').replace('.', '').replace(':', '').strip()
    
    # Try exact match first
    for vocab in vocabulary:
        if response_clean.lower() == vocab.lower():
            predicted_label = vocab
            confidence = 1.0
            break
    
    # If no exact match, try fuzzy matching
    if predicted_label is None:
        response_lower = response.lower()
        for vocab in vocabulary:
            vocab_lower = vocab.lower()
            patterns = [
                rf'\b{re.escape(vocab_lower)}\b',
                rf'"{re.escape(vocab_lower)}"',
                rf"'{re.escape(vocab_lower)}'",
            ]
            
            for pattern in patterns:
                if re.search(pattern, response_lower):
                    predicted_label = vocab
                    confidence = 0.9
                    break
            
            if predicted_label is not None:
                break
    
    # If still no match, it's a failure
    if predicted_label is None:
        predicted_label = "PARSE_FAILED"
        confidence = 0.0
    
    timing['total'] = tokenize_time + generate_time + decode_time
    timing['per_example'] = timing['total']
    
    return predicted_label, confidence, timing, raw_response


def classify_batch_with_zephyr(
    model,
    tokenizer,
    task: str,
    data_batch: List[str],
    vocabulary: List[str],
    device: str = 'cuda'
) -> Tuple[List[Tuple[str, float, Dict[str, float], str]], Dict[str, float]]:
    """
    Classify a batch of texts using Zephyr-7B.
    
    Returns:
        (List of (predicted_label, confidence, per_example_timing, raw_response), batch_timing)
    """
    batch_size = len(data_batch)
    
    # Create prompts
    prompts = [create_classification_prompt(task, data, vocabulary) for data in data_batch]
    
    # Time batch tokenization
    tokenize_start = time.perf_counter()
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_lengths = [len(tokenizer.encode(p, add_special_tokens=True)) for p in prompts]
    tokenize_time = time.perf_counter() - tokenize_start
    
    # Time batch generation
    generate_start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    generate_time = time.perf_counter() - generate_start
    
    # Time batch decoding
    decode_start = time.perf_counter()
    responses = []
    for i in range(batch_size):
        generated_tokens = outputs[i][input_lengths[i]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        responses.append(response)
    decode_time = time.perf_counter() - decode_start
    
    # Batch timing
    batch_timing = {
        'tokenization': tokenize_time,
        'generation': generate_time,
        'decoding': decode_time,
        'total': tokenize_time + generate_time + decode_time,
        'per_example': (tokenize_time + generate_time + decode_time) / batch_size,
        'batch_size': batch_size
    }
    
    # Parse all responses
    results = []
    for response in responses:
        raw_response = response
        predicted_label = None
        confidence = 0.0
        
        # Clean response
        response_clean = response.strip()
        for prefix in ['Label:', 'The label is:', 'Answer:']:
            if response_clean.startswith(prefix):
                response_clean = response_clean[len(prefix):].strip()
        
        response_clean = response_clean.strip('"\'').replace('.', '').replace(':', '').strip()
        
        # Try exact match
        for vocab in vocabulary:
            if response_clean.lower() == vocab.lower():
                predicted_label = vocab
                confidence = 1.0
                break
        
        # Try fuzzy match
        if predicted_label is None:
            response_lower = response.lower()
            for vocab in vocabulary:
                vocab_lower = vocab.lower()
                patterns = [
                    rf'\b{re.escape(vocab_lower)}\b',
                    rf'"{re.escape(vocab_lower)}"',
                    rf"'{re.escape(vocab_lower)}'",
                ]
                
                for pattern in patterns:
                    if re.search(pattern, response_lower):
                        predicted_label = vocab
                        confidence = 0.9
                        break
                
                if predicted_label is not None:
                    break
        
        if predicted_label is None:
            predicted_label = "PARSE_FAILED"
            confidence = 0.0
        
        # Create per-example timing
        example_timing = {
            'tokenization': batch_timing['tokenization'] / batch_size,
            'generation': batch_timing['generation'] / batch_size,
            'decoding': batch_timing['decoding'] / batch_size,
            'total': batch_timing['per_example'],
            'per_example': batch_timing['per_example']
        }
        
        results.append((predicted_label, confidence, example_timing, raw_response))
    
    return results, batch_timing


def compute_timing_statistics(timings: List[float]) -> Dict[str, float]:
    """Compute timing statistics in milliseconds."""
    if not timings:
        return {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0}
        
    timings_ms = [t * 1000 for t in timings]
    return {
        'min': np.min(timings_ms),
        'max': np.max(timings_ms),
        'mean': np.mean(timings_ms),
        'median': np.median(timings_ms),
        'std': np.std(timings_ms)
    }


def run_zephyr_benchmark(limit_examples: int = None, batch_size: int = 1, verbose: bool = False):
    """Run classification benchmark with Zephyr-7B."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print("ZEPHYR-7B CLASSIFICATION BENCHMARK")
    print("="*70)
    
    # Load model
    model, tokenizer = load_zephyr_model(device)
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    param_memory_gb = (total_params * 2) / (1024**3)  # fp16
    print(f"\nModel: Zephyr-7B")
    print(f"Parameters: {total_params/1e9:.1f}B")
    print(f"Memory (fp16): ~{param_memory_gb:.1f}GB")
    print(f"Batch size: {batch_size}")
    
    # Warm up
    print("\nWarming up model...")
    warmup_prompts = ["Classify this: test. Labels: A, B"] * min(3, batch_size)
    for _ in range(3):
        warmup_inputs = tokenizer(warmup_prompts[:batch_size], return_tensors="pt", 
                                padding=True, truncation=True).to(device)
        with torch.no_grad():
            _ = model.generate(**warmup_inputs, max_new_tokens=5, do_sample=False)
    print("Warmup complete.")
    
    # Test cases
    test_cases = TEST_CASES
    
    print(f"\nRunning tests")
    if limit_examples:
        print(f"Limiting to {limit_examples} examples per task")
    if verbose:
        print("Verbose mode: showing all examples")
    
    all_results = []
    batch_timings = {
        'tokenization': [],
        'generation': [],
        'decoding': [],
        'total': [],
        'per_example': []
    }
    per_example_timings = {
        'tokenization': [],
        'generation': [],
        'decoding': [],
        'total': []
    }
    task_summaries = []
    
    # Run tests
    for test_idx, test_case in enumerate(test_cases):
        print(f"\n{'='*50}")
        print(f"TASK: {test_case['name']}")
        print(f"{'='*50}")
        
        task = test_case["task"]
        vocabulary = test_case["vocabulary"]
        print(f"Task: {task[:80]}...")
        print(f"Labels: {vocabulary}")
        
        task_correct = 0
        task_parse_failures = 0
        task_per_example_timings = []
        
        # Process examples
        examples = test_case["examples"]
        if limit_examples:
            examples = examples[:limit_examples]
        
        example_counter = 0
        
        # Process in batches
        for batch_start in range(0, len(examples), batch_size):
            batch_end = min(batch_start + batch_size, len(examples))
            batch_examples = examples[batch_start:batch_end]
            batch_data = [ex[0] for ex in batch_examples]
            batch_expected = [ex[1] for ex in batch_examples]
            
            if len(batch_data) == 1:
                # Single example
                data, expected_label = batch_data[0], batch_expected[0]
                pred_label, confidence, timing, raw_response = classify_with_zephyr(
                    model, tokenizer, task, data, vocabulary, device
                )
                batch_results = [(pred_label, confidence, timing, raw_response)]
                batch_timing = {
                    'tokenization': timing['tokenization'],
                    'generation': timing['generation'],
                    'decoding': timing['decoding'],
                    'total': timing['total'],
                    'per_example': timing['per_example'],
                    'batch_size': 1
                }
            else:
                # Batch processing
                batch_results, batch_timing = classify_batch_with_zephyr(
                    model, tokenizer, task, batch_data, vocabulary, device
                )
            
            # Store batch timings
            for key in ['tokenization', 'generation', 'decoding', 'total']:
                batch_timings[key].append(batch_timing[key])
            batch_timings['per_example'].append(batch_timing['per_example'])
            
            # Process results
            for idx, ((pred_label, confidence, example_timing, raw_response), expected_label) in enumerate(
                zip(batch_results, batch_expected)
            ):
                data = batch_data[idx]
                example_counter += 1
                
                # Check correctness
                if pred_label == "PARSE_FAILED":
                    task_parse_failures += 1
                    is_correct = False
                else:
                    is_correct = pred_label == expected_label
                    if is_correct:
                        task_correct += 1
                
                # Store timings
                for key in ['tokenization', 'generation', 'decoding', 'total']:
                    per_example_timings[key].append(example_timing[key])
                task_per_example_timings.append(example_timing['total'])
                
                # Store result
                all_results.append({
                    "task": test_case["name"],
                    "data": data,
                    "expected": expected_label,
                    "predicted": pred_label,
                    "raw_response": raw_response,
                    "is_correct": is_correct,
                    "parse_failed": pred_label == "PARSE_FAILED",
                    "confidence": confidence,
                    "inference_time_ms": example_timing['total'] * 1000
                })
                
                # Print examples
                if verbose or example_counter <= 3:
                    result_str = "✓" if is_correct else "✗"
                    if pred_label == "PARSE_FAILED":
                        result_str = "✗ (PARSE)"
                    print(f"\n  Example {example_counter}: {result_str}")
                    print(f"  Text: \"{data[:60]}...\"" if len(data) > 60 else f"  Text: \"{data}\"")
                    print(f"  Expected: {expected_label}")
                    print(f"  Predicted: {pred_label}")
                    print(f"  Raw output: '{raw_response}'")
                    print(f"  Time: {example_timing['total']*1000:.1f}ms")
        
        # Task summary
        task_accuracy = task_correct / len(examples) * 100
        task_timing_stats = compute_timing_statistics(task_per_example_timings)
        task_summaries.append({
            "name": test_case["name"],
            "correct": task_correct,
            "total": len(examples),
            "accuracy": task_accuracy,
            "parse_failures": task_parse_failures,
            "timing_stats": task_timing_stats
        })
        
        print(f"\n  Task Score: {task_correct}/{len(examples)} ({task_accuracy:.1f}%)")
        if task_parse_failures > 0:
            print(f"  Parse failures: {task_parse_failures}")
        print(f"  Task Timing per example (ms): median={task_timing_stats['median']:.2f}")
    
    # Overall summary
    total_correct = sum(1 for r in all_results if r['is_correct'])
    total_parse_failures = sum(1 for r in all_results if r['parse_failed'])
    total_tests = len(all_results)
    overall_accuracy = total_correct / total_tests * 100 if total_tests > 0 else 0
    
    # Print scorecard
    print("\n" + "="*100)
    print("FINAL SCORECARD - ERRORS ONLY")
    print("="*100)
    print(f"{'Task':<30} {'Expected':<20} {'Predicted':<20} {'Result':<8}")
    print("-" * 90)
    
    error_count = 0
    for result in all_results:
        if not result["is_correct"]:
            error_count += 1
            result_str = "✗ (PARSE)" if result["parse_failed"] else "✗"
            print(f"{result['task']:<30} {result['expected']:<20} {result['predicted']:<20} {result_str:<8}")
    
    if error_count == 0:
        print("  No errors - all predictions correct!")
    
    print("-" * 90)
    print(f"{'TOTAL CORRECT':<30} {'':<20} {'':<20} {total_correct}/{total_tests}")
    print(f"{'OVERALL ACCURACY':<30} {'':<20} {'':<20} {overall_accuracy:.1f}%")
    print("="*100)
    
    # Timing statistics
    print("\n" + "="*70)
    print(f"TIMING STATISTICS (milliseconds) - Batch Size: {batch_size}")
    print("="*70)
    
    print("\nPER-EXAMPLE TIMING:")
    if per_example_timings['total']:
        stats = compute_timing_statistics(per_example_timings['total'])
        print(f"  Median: {stats['median']:.2f} ms")
        print(f"  Mean:   {stats['mean']:.2f} ms")
        print(f"  Min:    {stats['min']:.2f} ms")
        print(f"  Max:    {stats['max']:.2f} ms")
    
    # Comparison with DSRU
    print("\n" + "="*70)
    print("COMPARISON WITH DSRU")
    print("="*70)
    
    if per_example_timings['total']:
        median_time = np.median([t * 1000 for t in per_example_timings['total']])
        print(f"DSRU median latency: 1.30ms")
        print(f"Zephyr-7B median latency: {median_time:.2f}ms")
        print(f"DSRU is {median_time/1.30:.1f}x faster")
        
    print(f"\nDSRU accuracy: 77.7%")
    print(f"Zephyr-7B accuracy: {overall_accuracy:.1f}%")
    
    if overall_accuracy > 77.7:
        print(f"Zephyr-7B has {overall_accuracy - 77.7:.1f}% higher accuracy")
    else:
        print(f"DSRU has {77.7 - overall_accuracy:.1f}% higher accuracy")
    
    print("="*70)
    
    # Save results
    results_data = {
        "model": "zephyr-7b",
        "batch_size": batch_size,
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_tests": total_tests,
        "parse_failures": total_parse_failures,
        "timing_stats": {
            "per_example": compute_timing_statistics(per_example_timings['total']) if per_example_timings['total'] else {}
        },
        "task_summaries": task_summaries
    }
    
    with open(f"zephyr_benchmark_batch{batch_size}.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: zephyr_benchmark_batch{batch_size}.json")


def main():
    parser = argparse.ArgumentParser(description='Benchmark Zephyr-7B on classification tasks')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit examples per task')
    parser.add_argument('--batch', type=int, default=1,
                        help='Batch size for inference (default: 1)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with 3 examples per task')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show all examples instead of just first 3')
    
    args = parser.parse_args()
    
    if args.quick:
        args.limit = 3
    
    run_zephyr_benchmark(
        limit_examples=args.limit, 
        batch_size=args.batch,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
