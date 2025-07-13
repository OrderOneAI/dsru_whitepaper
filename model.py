#!/usr/bin/env python3
"""
Scaled BGE Vector Reasoning Engine - With Vocabulary Embeddings
Architecture:
- BGE-Large Embeddings: 1024 dimensions
- 3 separate input projections (task, data, vocab) → 4096 each
- 14 layers with 8192 hidden dimensions each
- ~300M parameters with improved semantic targeting
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class ScaledVectorReasoningEngine(nn.Module):
    """
    Scaled Vector Reasoning Engine with Vocabulary Embeddings
    
    Enhanced architecture with explicit vocabulary guidance:
    - Inputs: Task vector + Data vector + Vocabulary vector (1024 dims each)
    - Processing: 3 separate projections → concatenated → deep reasoning
    - Output: Answer vector (1024 dims, BGE-large space)
    """
    
    def __init__(self, 
                 vector_dim: int = 1024,        # BGE-large dimension
                 projection_dim: int = 4096,    # Dimension for each input projection
                 hidden_dim: int = 8192,        # Hidden layer dimension
                 num_layers: int = 8):          # Number of reasoning layers
        """
        Initialize scaled vector reasoning engine with vocabulary support.
        
        Args:
            vector_dim: BGE-large dimension (1024)
            projection_dim: Dimension for each input projection (4096)
            hidden_dim: Hidden layer dimension (8192)
            num_layers: Number of reasoning layers (8)
        """
        super().__init__()
        
        self.vector_dim = vector_dim
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Sparse input layer - each input projects to its own region
        sparse_dim = projection_dim * 3  # 12288 total
        
        # Create sparse projection matrix 
        # We manually handle sparsity to ensure each input (task/data/vocab) 
        # projects to its own dedicated subspace without interference
        self.sparse_layer = nn.Parameter(torch.zeros(vector_dim * 3, sparse_dim))
        
        # Initialize only the relevant parts
        # Task: [0:1024] → [0:4095]
        # Data: [1024:2048] → [4096:8191]
        # Vocab: [2048:3072] → [8192:12287]
        nn.init.xavier_uniform_(self.sparse_layer[0:vector_dim, 0:projection_dim])
        nn.init.xavier_uniform_(self.sparse_layer[vector_dim:vector_dim*2, projection_dim:projection_dim*2])
        nn.init.xavier_uniform_(self.sparse_layer[vector_dim*2:vector_dim*3, projection_dim*2:projection_dim*3])
        
        # First hidden layer: sparse output → hidden
        self.input_layer = nn.Linear(sparse_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Hidden reasoning layers with skip connections
        self.hidden_layers = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()
        
        for i in range(num_layers - 2):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_norms.append(nn.LayerNorm(hidden_dim))
        
        # Output projection: hidden → vector_dim
        self.output_layer = nn.Linear(hidden_dim, vector_dim)
        self.output_norm = nn.LayerNorm(vector_dim)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        param_millions = total_params / 1_000_000
        
        logger.info(f"Scaled Vector Reasoning Engine with Vocabulary initialized:")
        logger.info(f"  Vector dimension: {vector_dim}")
        logger.info(f"  Projection dimension: {projection_dim} (per input)")
        logger.info(f"  Hidden dimension: {hidden_dim}")
        logger.info(f"  Number of layers: {num_layers}")
        logger.info(f"  Total parameters: {total_params:,} ({param_millions:.1f}M)")
        logger.info(f"  Architecture: 3 input projections → deep reasoning → output")
        
        # Log parameter breakdown
        sparse_params = vector_dim * projection_dim * 3  # Only non-zero weights count
        input_params = projection_dim * 3 * hidden_dim
        hidden_params = (num_layers - 2) * hidden_dim * hidden_dim
        output_params = hidden_dim * vector_dim
        
        logger.info(f"  Parameter breakdown:")
        logger.info(f"    Sparse layer: {sparse_params:,} ({sparse_params/1_000_000:.1f}M)")
        logger.info(f"    Input layer: {input_params:,} ({input_params/1_000_000:.1f}M)")
        logger.info(f"    Hidden layers: {hidden_params:,} ({hidden_params/1_000_000:.1f}M)")
        logger.info(f"    Output layer: {output_params:,} ({output_params/1_000_000:.1f}M)")
    
    def forward(self, 
                task_vectors: torch.Tensor, 
                data_vectors: torch.Tensor,
                vocab_vectors: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Task + Data + Vocabulary → Answer vectors
        
        The vocabulary embedding provides explicit guidance about the
        valid answer space, enabling more precise reasoning.
        
        Args:
            task_vectors: [batch_size, vector_dim] - Task embeddings
            data_vectors: [batch_size, vector_dim] - Data embeddings
            vocab_vectors: [batch_size, vector_dim] - Vocabulary embeddings
            
        Returns:
            answer_vectors: [batch_size, vector_dim] - Answer in BGE space
        """
        # Concatenate all inputs
        x = torch.cat([task_vectors, data_vectors, vocab_vectors], dim=1)
        
        # Sparse projection - manual matrix multiply with our sparse weights
        x = torch.matmul(x, self.sparse_layer)
        x = F.gelu(x)
        
        # First hidden layer
        x = self.input_layer(x)
        x = self.input_norm(x)
        x = F.gelu(x)
        
        # Hidden layers with skip connections
        for i in range(self.num_layers - 2):
            residual = x
            x = self.hidden_layers[i](x)
            x = self.hidden_norms[i](x)
            x = F.gelu(x)
            x = x + residual
        
        # Output projection
        x = self.output_layer(x)
        x = self.output_norm(x)
        
        # L2 normalize to unit vectors
        answer_vectors = F.normalize(x, p=2, dim=1)
        
        return answer_vectors
    
    def compute_semantic_loss(self, 
                             task_vectors: torch.Tensor, 
                             data_vectors: torch.Tensor,
                             vocab_vectors: torch.Tensor,
                             target_answer_vectors: torch.Tensor) -> torch.Tensor:
        """
        Compute semantic loss using cosine similarity.
        
        Args:
            task_vectors: [batch_size, vector_dim]
            data_vectors: [batch_size, vector_dim]
            vocab_vectors: [batch_size, vector_dim]
            target_answer_vectors: [batch_size, vector_dim]
            
        Returns:
            loss: Scalar loss tensor (1 - cosine_similarity)
        """
        predicted_answer_vectors = self.forward(task_vectors, data_vectors, vocab_vectors)
        cosine_sim = F.cosine_similarity(predicted_answer_vectors, target_answer_vectors, dim=1)
        loss = (1 - cosine_sim).mean()
        return loss
    
    def compute_batch_accuracy(self, 
                              task_vectors: torch.Tensor, 
                              data_vectors: torch.Tensor,
                              vocab_vectors: torch.Tensor,
                              target_answer_vectors: torch.Tensor,
                              threshold: float = 0.9) -> float:
        """
        Compute accuracy based on cosine similarity threshold.
        
        Args:
            task_vectors: [batch_size, vector_dim]
            data_vectors: [batch_size, vector_dim]
            vocab_vectors: [batch_size, vector_dim]
            target_answer_vectors: [batch_size, vector_dim]
            threshold: Similarity threshold for "correct"
            
        Returns:
            accuracy: Fraction of examples above threshold
        """
        with torch.no_grad():
            task_vectors = task_vectors.detach()
            data_vectors = data_vectors.detach()
            vocab_vectors = vocab_vectors.detach()
            target_answer_vectors = target_answer_vectors.detach()
            
            predicted_answer_vectors = self.forward(task_vectors, data_vectors, vocab_vectors)
            cosine_sim = F.cosine_similarity(predicted_answer_vectors, target_answer_vectors, dim=1)
            correct = (cosine_sim > threshold).float()
            accuracy = correct.mean().item()
        
        return accuracy
    
    def compute_multi_threshold_accuracy(self,
                                       task_vectors: torch.Tensor,
                                       data_vectors: torch.Tensor,
                                       vocab_vectors: torch.Tensor,
                                       target_answer_vectors: torch.Tensor,
                                       thresholds: List[float] = [0.025, 0.05, 0.1, 0.15]) -> dict:
        """
        Compute accuracy at multiple cosine distance thresholds.
        
        Args:
            task_vectors: [batch_size, vector_dim]
            data_vectors: [batch_size, vector_dim]
            vocab_vectors: [batch_size, vector_dim]
            target_answer_vectors: [batch_size, vector_dim]
            thresholds: List of distance thresholds (not similarity!)
            
        Returns:
            accuracies: Dict mapping threshold to accuracy
        """
        with torch.no_grad():
            predicted_answer_vectors = self.forward(task_vectors, data_vectors, vocab_vectors)
            cosine_sim = F.cosine_similarity(predicted_answer_vectors, target_answer_vectors, dim=1)
            cosine_distance = 1 - cosine_sim
            
            accuracies = {}
            for threshold in thresholds:
                correct = (cosine_distance <= threshold).float()
                accuracies[threshold] = correct.mean().item()
        
        return accuracies
    
    def get_intermediate_representations(self, 
                                       task_vectors: torch.Tensor,
                                       data_vectors: torch.Tensor,
                                       vocab_vectors: torch.Tensor) -> List[torch.Tensor]:
        """
        Get intermediate reasoning steps for analysis.
        
        Returns representations after each layer, showing how the model
        progressively refines its answer based on the vocabulary guidance.
        """
        intermediate_vectors = []
        
        # Concatenate inputs
        x = torch.cat([task_vectors, data_vectors, vocab_vectors], dim=1)
        
        # Sparse projection
        x = torch.matmul(x, self.sparse_layer)
        x = F.gelu(x)
        intermediate_vectors.append(x.clone())
        
        # First hidden layer
        x = F.gelu(self.input_norm(self.input_layer(x)))
        intermediate_vectors.append(x.clone())
        
        # Hidden layers
        for i in range(self.num_layers - 2):
            residual = x
            x = F.gelu(self.hidden_norms[i](self.hidden_layers[i](x)))
            x = x + residual
            intermediate_vectors.append(x.clone())
        
        # Output
        x = self.output_norm(self.output_layer(x))
        x = F.normalize(x, p=2, dim=1)
        intermediate_vectors.append(x)
        
        return intermediate_vectors
