# Direct Semantic Reasoning Unit (DSRU) - Reference Implementation

This repository contains the reference implementation of the Direct Semantic Reasoning Unit (DSRU), a novel neural architecture that performs reasoning in constant time O(1) over semantic embeddings.

## Overview

DSRU processes entire concepts and tasks in a single forward pass without tokens or attention mechanisms, achieving:
- **93x faster inference** than comparably performing transformer models
- **1ms inference time** for complex reasoning tasks
- **77.7% average accuracy** across 13 diverse classification tasks

Note that this is NOT a generative model, but a predictive one. The key difference between this and typical/earlier predictive models is that it seems capable of reasoning in a fashion that's somewhat similar to LLMs (achieving 80% on a logical entailment task after an hour of training on a 4060 Ti), and is promptable. The white paper goes into more detail about the model's workings and capabilities, while this repository demonstrates its application in a promptable classifier.

## Key Innovation

Instead of processing sequences token-by-token with quadratic attention complexity, DSRU operates directly on semantic embeddings, treating them as complete thoughts that can be transformed through learned reasoning operations.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- sentence-transformers
- numpy

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Download the model checkpoint from [Google Drive](https://drive.google.com/file/d/1oZarHzA7PwSij6aBGOQaEHCnB100St3H/view?usp=sharing) and place it as `model.pt`

## Quick Start

```python
python inference.py --batch 32
```

This will run the inference benchmarks on the included test cases with a batch size of 32.

## Model Architecture

The DSRU uses a three-input architecture:
- **Task embedding**: What operation to perform
- **Data embedding**: The input to process  
- **Vocabulary embedding**: Valid output space

These are projected through sparse layers into a deep reasoning network with:
- 1024-dimensional input/output (BGE-large embedding space)
- 8192-dimensional hidden layers
- 14 total layers with residual connections
- ~1.09B parameters

## Files

- `inference.py` - Main inference script with timing benchmarks
- `model.py` - DSRU model implementation
- `vocabulary_helpers.py` - Embedding utilities
- `inference_questions.py` - Test cases and benchmarks
- `model.pt` - Trained model checkpoint [Google Drive](https://drive.google.com/file/d/1oZarHzA7PwSij6aBGOQaEHCnB100St3H/view?usp=sharing)

## Performance

On a single GPU, DSRU achieves:
- **Throughput**: 93x higher than Zephyr-7B (1.3ms per example when batched)
- **Latency**: 30ms single-request
- **Accuracy**: 77.7% average across diverse tasks

## License

Free for personal use, including personal, unpaid academic use, such as thesis papers. All institutional or commercial use is prohibited. Further licensing details will be forthcoming.

## Patent Notice

The DSRU architecture and training methods are patent pending. Personal experimentation is permitted.

## Paper

For technical details and theoretical background, see the accompanying white paper: "Direct Semantic Reasoning Unit: The O(1) AI Primitive That Reasons In Latent Space"
