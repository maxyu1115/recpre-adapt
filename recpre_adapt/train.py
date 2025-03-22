from typing import Callable
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
import sys
import logging

# Add the parent directory to the Python path to make recpre importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recpre.raven_modeling_minimal import RavenForCausalLM, CausalSelfAttention
from recpre_adapt.data_loaders.red_pajama import RedPajamaPMD
from recpre_adapt.data_loaders.gsm8k import GSM8K
from recpre_adapt.raven_exit_model import RavenExitModel

EPSILON = 1e-10

# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def update_huggingface_implementation(model):
    """This function selectively updates function implementations in the huggingface model."""
    import types
    model.iterate_forward = types.MethodType(RavenForCausalLM.iterate_forward, model)
    # for name, module in model.named_modules():
    #     if module.__class__.__name__ == "CausalSelfAttention":
    #         module.forward = types.MethodType(CausalSelfAttention.forward, module)


def generate_causal_mask(seq_len: int, device=None):
    """
    Generate a causal attention mask compatible with transformer models.
    Returns a mask where 1 means "attend" and 0 means "don't attend".
    """
    # Return None, since we don't need a mask and pass causal=True to the model
    return None

def calculate_scores(
    model: RavenForCausalLM,
    latents: list[torch.Tensor],
    score_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> list[torch.Tensor]:
    with torch.no_grad():
        last_latent = latents[-1]
        target: torch.Tensor = torch.softmax(model.predict_from_latents(last_latent).logits, dim=-1) # type: ignore
        scores = []
        for latent in latents:
            latent_prob: torch.Tensor = torch.softmax(model.predict_from_latents(latent).logits, dim=-1) # type: ignore
            score = score_func(target, latent_prob)
            scores.append(score)
    return scores


def score_ce_top_k(target: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    # shape of target and predicted: batch_size, seq_len, vocab_size

    # Get top k values and indices for both target and predicted logits
    k = 10  # Can be adjusted as needed
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
    
    return matches * -1



def compute_loss(
    model: RavenForCausalLM,
    exit_model: RavenExitModel,
    x: torch.Tensor,
    num_steps: int,
    discount: float,
    target_identity_policy: torch.Tensor,
):
    with torch.no_grad():
        model.forward(x, attention_mask=None, num_steps=torch.tensor((num_steps,)))
        assert len(model.latents) == num_steps + 1
        latents: list[torch.Tensor] = model.latents
        scores = calculate_scores(model, latents, score_cross_entropy)
        # scores = calculate_scores(model, latents, score_ce_top_k)
        # scores = calculate_scores(model, latents, count_matching_topk)
        # for i in range(num_steps):
        #     print(f"Step {i}, Score: {scores[i].mean().item()}")
        # return

    # loss for the last step. shape: batch_size, seq_len
    # expected_returns = [torch.zeros_like(scores[num_steps])] * (num_steps + 1)
    # expected_returns[num_steps] = scores[num_steps]
    expected_loss = scores[num_steps]
    expected_steps = torch.ones_like(scores[num_steps]) * num_steps

    # Iterate backwards from numsteps - 1 to 1, since we need the next reward to calculate the current one
    for i in range(num_steps - 1, 0, -1):
        # latent dimensions: batch_size, seq_len, n_embd
        # policy dimensions: batch_size, seq_len, 2
        policy = exit_model.forward(latents[i - 1], latents[i])
        # 0 means we exit, 1 means we continue
        # NOTE: scores here are (0, inf), and we penalize an additional cost for each extra
        # expected_returns[i] = policy[:, :, 0] * (scores[i] + cost * i) + policy[:, :, 1] * expected_returns[i + 1].detach()
        # expected_return = policy[:, :, 0] * (scores[i] + cost * i) + policy[:, :, 1] * expected_return
        expected_loss = policy[:, :, 0] * (scores[i]) * (2 - discount ** i) + policy[:, :, 1] * expected_loss
        with torch.no_grad():
            expected_steps = policy[:, :, 0] * i + policy[:, :, 1] * expected_steps

        # Calculate policy gradient loss
        # # We want to maximize expected reward, so we minimize negative expected reward
        # # policy[:, :, 0] is probability of exiting, policy[:, :, 1] is probability of continuing
        # action_probs = torch.stack([policy[:, :, 0], policy[:, :, 1]], dim=1)

        # # Calculate loss - negative expected reward
        # loss = -torch.mean(r * torch.log(action_probs + 1e-10))

    identity_loss = torch.zeros((x.shape[0], x.shape[1]), device=model.device)
    for i in range(num_steps):
        identity_policy = exit_model.forward(latents[i], latents[i])
        identity_loss += -torch.sum(target_identity_policy * torch.log(identity_policy + EPSILON), dim=-1)

    # Backpropagate and update model
    identity_loss = identity_loss / num_steps
    return expected_loss, identity_loss, expected_steps

def train(
    model: RavenForCausalLM,
    exit_model: RavenExitModel,
    redpajama_pmd: RedPajamaPMD,
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    batch_size: int,
    seq_len: int,
    training_epochs: int = 100,
    discount: float = 0.99,
    identity_loss_weight: float = 0.1,
    verbose: bool = True,
):
    model.eval()

    target_identity_policy = torch.zeros((batch_size, seq_len - 1, 2), device=model.device)
    target_identity_policy[:, :, 0] = 1.0

    x_val = []
    for i in range(10):
        x, _ = redpajama_pmd.get_batch("train")
        x_val.append(x)

    for epoch in range(training_epochs):
        x, _ = redpajama_pmd.get_batch("train")

        # Train the exit model using the calculated rewards
        optimizer.zero_grad()

        expected_loss, identity_loss, expected_steps = compute_loss(model, exit_model, x, num_steps, discount, target_identity_policy)
        avg_loss = torch.mean(expected_loss + identity_loss_weight * identity_loss)

        avg_loss.backward()
        optimizer.step()

        # if verbose and (epoch < 30 or epoch % 10 == 0):
        if verbose and (epoch < 30):
            logging.info(f"Epoch {epoch}, Loss: {avg_loss.item()}, Expected Steps: {torch.mean(expected_steps).item()} std: {torch.std(expected_steps, dim=-1)} max: {torch.max(expected_steps)}")

        if epoch % 10 == 0:
            with torch.no_grad():
                val_losses = []
                val_identity_losses = []
                val_expected_steps = []
                for x in x_val:
                    expected_loss, identity_loss, expected_steps = compute_loss(model, exit_model, x, num_steps, discount, target_identity_policy)
                    val_losses.append(torch.mean(expected_loss + identity_loss_weight * identity_loss))
                    val_identity_losses.append(torch.mean(identity_loss))
                    val_expected_steps.append(expected_steps)
                val_loss = torch.mean(torch.stack(val_losses))
                val_identity_loss = torch.mean(torch.stack(val_identity_losses))
                val_expected_steps = torch.stack(val_expected_steps)
                logging.info(f"Epoch {epoch}, Validation Loss: {val_loss.item()}, Identity Loss: {val_identity_loss.item()}, Expected Steps: {torch.mean(val_expected_steps).item()} std: {torch.std(val_expected_steps, dim=-1).data} max: {torch.max(val_expected_steps).item()}")
        if epoch % 100 == 99:
            # Save the model
            torch.save(exit_model.state_dict(), f"exit_model_{epoch}.pt")


def main(num_steps: int = 32, batch_size: int = 2, seq_len: int = 1024):
    model: RavenForCausalLM = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")

    update_huggingface_implementation(model)

    # model.to("cuda", dtype=torch.bfloat16) # type: ignore
    model.to("cuda", dtype=torch.bfloat16) # type: ignore
    model.save_latents = True

    torch.manual_seed(42)
    # torch.manual_seed(1)

    redpajama_pmd = RedPajamaPMD(model.device, tokenizer, batch_size, seq_len)
    # gsm8k_pmd = GSM8K(model.device, tokenizer, batch_size, seq_len)
    exit_model = RavenExitModel(model.config)
    exit_model.to(dtype=torch.float32, device=model.device)

    optimizer = torch.optim.AdamW(exit_model.parameters(), lr=3e-5)

    # Add a non-zero cost to encourage efficiency
    train(model, exit_model, redpajama_pmd, optimizer, num_steps=num_steps, 
          batch_size=batch_size, seq_len=seq_len, training_epochs=1000,
          discount=0.99, identity_loss_weight=0.05, verbose=True)


if __name__ == "__main__":
    main(batch_size=2, seq_len=1024)
