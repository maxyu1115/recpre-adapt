import torch
from typing import Callable, Optional

EPSILON = 1e-10


SCORE_FUNC = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def calculate_scores(
    probs: list[torch.Tensor],
    score_func: SCORE_FUNC,
) -> list[torch.Tensor]:
    with torch.no_grad():
        target: torch.Tensor = probs[-1]
        scores = [score_func(target, prob) for prob in probs]
    return scores


def calculate_optimal_scores(
    probs: list[torch.Tensor],
    score_func: SCORE_FUNC,
    penalty: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        scores = calculate_scores(probs, score_func)
        scores = [score * (penalty ** i) for i, score in enumerate(scores)]
        scores = torch.stack(scores, dim=-2)
        optimal_scores, optimal_indices = torch.min(scores, dim=-2)
    return optimal_scores, optimal_indices


def score_perplexity(target: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    return torch.exp(score_cross_entropy(target, predicted))

def score_ce_top_k(target: torch.Tensor, predicted: torch.Tensor, k: int = 10) -> torch.Tensor:
    # shape of target and predicted: batch_size, seq_len, vocab_size

    # Get top k values and indices for both target and predicted logits
    target_topk = torch.topk(target, k, dim=-1)
    predicted_topk = torch.topk(predicted, k, dim=-1)

    # Create masks for top k indices
    target_mask = torch.zeros_like(target).scatter_(-1, target_topk.indices, 1.0)
    predicted_mask = torch.zeros_like(predicted).scatter_(-1, predicted_topk.indices, 1.0)

    # Zero out non-top k values
    target_filtered = target * target_mask
    log_predicted_filtered = torch.log(predicted + EPSILON) * predicted_mask

    # Calculate cross entropy loss between filtered values
    cross_entropy = -torch.sum(target_filtered * log_predicted_filtered, dim=-1)
    return cross_entropy


def score_cross_entropy(target: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    """
    Calculate cross-entropy between two probability distributions.
    
    Args:
        target: Target probability distribution (batch_size, seq_len, vocab_size)
        predicted: Predicted probability distribution (batch_size, seq_len, vocab_size)
        
    Returns:
        Cross-entropy loss (batch_size, seq_len)
    """
    assert target.shape == predicted.shape

    # Proper cross-entropy between distributions: -sum(target * log(predicted))
    log_predicted = torch.log(predicted + EPSILON)
    cross_entropy = -torch.sum(target * log_predicted, dim=-1)

    return cross_entropy

def score_ce_clamped(target: torch.Tensor, predicted: torch.Tensor, cutoff: float = 0.01) -> torch.Tensor:
    """
    Calculate the squared cross-entropy between two probability distributions.
    
    Args:
        target: Target probability distribution (batch_size, seq_len, vocab_size)
        predicted: Predicted probability distribution (batch_size, seq_len, vocab_size)
        
    Returns:
        Squared cross-entropy loss (batch_size, seq_len)
    """
    assert target.shape == predicted.shape

    # target = torch.softmax(target * 100, dim=-1)
    # predicted = torch.softmax(predicted * 100, dim=-1)

    # target = (target ** 2) / (target ** 2).sum(dim=-1, keepdim=True)
    # predicted = (predicted ** 2) / (predicted ** 2).sum(dim=-1, keepdim=True)

    target = torch.where(target < cutoff, torch.zeros_like(target), target)
    predicted = torch.where(predicted < cutoff, torch.zeros_like(predicted), predicted)

    log_predicted = torch.log(predicted + EPSILON)
    cross_entropy = -torch.sum(target * log_predicted, dim=-1)

    return cross_entropy

def score_prob_distance(target: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    return torch.sum((target - predicted) ** 2, dim=-1)

def count_matching_topk(target: torch.Tensor, predicted: torch.Tensor, k: int = 10) -> torch.Tensor:
    """
    Count the number of matching tokens in the top-k predictions between target and predicted distributions.
    
    Args:
        target: Target probability distribution (batch_size, seq_len, vocab_size)
        predicted: Predicted probability distribution (batch_size, seq_len, vocab_size)
        k: Number of top tokens to consider
        
    Returns:
        Tensor containing the count of matching tokens (batch_size, seq_len)
    """
    # Get top k indices for both target and predicted logits
    target_topk = torch.topk(target, k, dim=-1).indices  # (batch_size, seq_len, k)
    predicted_topk = torch.topk(predicted, k, dim=-1).indices  # (batch_size, seq_len, k)
    
    # Create a one-hot representation of target_topk
    batch_size, seq_len, _ = target.shape
    target_mask = torch.zeros_like(target).scatter_(-1, target_topk, 1.0)
    
    # Count matches by checking which predicted top-k tokens are in target top-k
    matches = torch.zeros((batch_size, seq_len), device=target.device)
    for i in range(k):
        # For each token in predicted top-k, check if it's in target top-k
        token_indices = predicted_topk[:, :, i].unsqueeze(-1)  # (batch_size, seq_len, 1)
        matches += target_mask.gather(-1, token_indices).squeeze(-1)  # (batch_size, seq_len)
    
    return matches

def score_matching_top135(target: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    top_1_matches = count_matching_topk(target, predicted, 1)
    top_3_matches = count_matching_topk(target, predicted, 3)
    top_5_matches = count_matching_topk(target, predicted, 5)
    return 4 - (top_1_matches + top_3_matches / 3.0 + top_5_matches / 5.0)


def get_score_func(score_func_name: str) -> SCORE_FUNC:
    if score_func_name == "ce":
        score_func = score_cross_entropy
    elif score_func_name == "clamped_ce_0.01":
        score_func = lambda target, predicted: score_ce_clamped(target, predicted, 0.01)
    elif score_func_name == "matching_top135":
        score_func = score_matching_top135
    elif score_func_name == "matching_top1":
        score_func = lambda target, predicted: 1 - count_matching_topk(target, predicted, 1)
    elif score_func_name == "ce_top10":
        score_func = lambda target, predicted: score_ce_top_k(target, predicted, 10)
    elif score_func_name == "perplexity":
        score_func = score_perplexity
    elif score_func_name == "prob_distance":
        score_func = score_prob_distance
    elif score_func_name == "ce_matching_top135":
        score_func = lambda target, predicted: score_cross_entropy(target, predicted) * score_matching_top135(target, predicted)
    else:
        raise ValueError(f"Invalid score function: {score_func_name}")
    return score_func

