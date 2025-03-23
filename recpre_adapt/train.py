import os
import sys
import logging
import json
from typing import Callable, Literal
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add the parent directory to the Python path to make recpre importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recpre.raven_modeling_minimal import RavenForCausalLM, CausalSelfAttention
from recpre_adapt.data_loaders import PoorMansDataLoaderBase
from recpre_adapt.data_loaders.red_pajama import RedPajamaPMD
from recpre_adapt.data_loaders.gsm8k import GSM8K
from recpre_adapt.data_loaders.math_pile import MathPilePMD
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
    id_sampling_interval: int,
    is_eval: bool = False,
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
    expected_true_loss = scores[num_steps]

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
            if is_eval:
                expected_true_loss = policy[:, :, 0] * scores[i] + policy[:, :, 1] * expected_true_loss

    identity_loss = torch.zeros((x.shape[0], x.shape[1]), device=model.device)
    for i in range(num_steps // id_sampling_interval):
        identity_policy = exit_model.forward(latents[i * id_sampling_interval], latents[i * id_sampling_interval])
        identity_loss += -torch.sum(target_identity_policy * torch.log(identity_policy + EPSILON), dim=-1)

    # Backpropagate and update model
    identity_loss = identity_loss / (num_steps // id_sampling_interval)
    return expected_loss, identity_loss, expected_steps, expected_true_loss

def train(
    model: RavenForCausalLM,
    exit_model: RavenExitModel,
    pmd: PoorMansDataLoaderBase,
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    batch_size: int,
    seq_len: int,
    training_epochs: int = 100,
    discount: float = 0.99,
    identity_loss_weight: float = 0.1,
    id_sampling_interval: int = 1,
    verbose: bool = True,
):
    model.eval()

    target_identity_policy = torch.zeros((batch_size, seq_len - 1, 2), device=model.device)
    target_identity_policy[:, :, 0] = 1.0

    x_val = []
    for i in range(10):
        x, _ = pmd.get_batch("train")
        x_val.append(x)

    for epoch in range(training_epochs):
        x, _ = pmd.get_batch("train")

        # Train the exit model using the calculated rewards
        optimizer.zero_grad()

        expected_loss, identity_loss, expected_steps, _ = compute_loss(model, exit_model, x, num_steps, discount, target_identity_policy, id_sampling_interval)
        avg_loss = torch.mean(expected_loss + identity_loss_weight * identity_loss)

        avg_loss.backward()
        optimizer.step()

        if epoch < 30 or (verbose and epoch % 10 == 0):
            logging.info(f"Epoch {epoch}, Loss: {avg_loss.item()}, "\
                f"Expected Steps: {torch.mean(expected_steps).item()}"\
                f" std: {torch.std(expected_steps, dim=-1)} "\
                f"max: {torch.max(expected_steps)}")

        if epoch % 10 == 0:
            with torch.no_grad():
                val_losses = []
                val_identity_losses = []
                val_expected_steps = []
                val_true_losses = []
                for x in x_val:
                    expected_loss, identity_loss, expected_steps, true_loss = compute_loss(model, exit_model, x, num_steps, discount, target_identity_policy, id_sampling_interval, is_eval=True)
                    val_losses.append(torch.mean(expected_loss + identity_loss_weight * identity_loss))
                    val_identity_losses.append(torch.mean(identity_loss))
                    val_expected_steps.append(expected_steps)
                    val_true_losses.append(true_loss)
                val_loss = torch.mean(torch.stack(val_losses))
                val_identity_loss = torch.mean(torch.stack(val_identity_losses))
                val_expected_steps = torch.stack(val_expected_steps)
                val_true_loss = torch.mean(torch.stack(val_true_losses))
                logging.info(f"Epoch {epoch}, True Loss: {val_true_loss.item()}, "\
                    f"Validation Loss: {val_loss.item()}, "\
                    f"Identity Loss: {val_identity_loss.item()}, "\
                    f"Expected Steps: {torch.mean(val_expected_steps).item()}"\
                    f" std: {torch.std(val_expected_steps, dim=-1)}"\
                    f" max: {torch.max(val_expected_steps)}")
        if epoch % 100 == 99:
            # Save the model
            torch.save(exit_model.state_dict(), f"exit_model_{epoch}.pt")


def main(
    dataset: Literal["red_pajama", "gsm8k", "math_pile"],
    num_steps: int = 32,
    batch_size: int = 2,
    seq_len: int = 1024
):
    torch_seed = 42
    discount = 0.99
    identity_loss_weight = 0.05
    id_sampling_interval = 1
    training_epochs = 1000
    learning_rate = 3e-5
    model: RavenForCausalLM = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")

    update_huggingface_implementation(model)

    # model.to("cuda", dtype=torch.bfloat16) # type: ignore
    model.to("cuda", dtype=torch.bfloat16) # type: ignore
    model.save_latents = True

    torch.manual_seed(torch_seed)

    if dataset == "red_pajama":
        pmd = RedPajamaPMD(model.device, tokenizer, batch_size, seq_len)
    elif dataset == "gsm8k":
        pmd = GSM8K(model.device, tokenizer, batch_size, seq_len)
    elif dataset == "math_pile":
        pmd = MathPilePMD(model.device, tokenizer, batch_size, seq_len)
    else:
        raise ValueError(f"Invalid dataset: {dataset}")
    exit_model = RavenExitModel(model.config)
    exit_model.to(dtype=torch.float32, device=model.device)

    optimizer = torch.optim.AdamW(exit_model.parameters(), lr=learning_rate)
    
    with open("training_params.json", "w") as f:
        json.dump({
            "dataset": dataset,
            "num_steps": num_steps,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "training_epochs": training_epochs,
            "learning_rate": learning_rate,
            "discount": discount,
            "identity_loss_weight": identity_loss_weight,
            "identity_sampling_interval": id_sampling_interval,
            "torch_seed": torch_seed,
        }, f)

    # Add a non-zero cost to encourage efficiency
    train(model, exit_model, pmd, optimizer, num_steps=num_steps, 
          batch_size=batch_size, seq_len=seq_len, training_epochs=training_epochs,
          discount=discount, identity_loss_weight=identity_loss_weight, id_sampling_interval=id_sampling_interval, verbose=False)


if __name__ == "__main__":
    main(dataset="red_pajama", batch_size=2, seq_len=512, num_steps=64)
