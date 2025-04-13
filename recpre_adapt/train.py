import os
import sys
import logging
import json
from typing import Callable, Literal
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import math

# Add the parent directory to the Python path to make recpre importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recpre.raven_modeling_minimal import RavenForCausalLM, CausalSelfAttention
from recpre_adapt.data_loaders import PoorMansDataLoaderBase
from recpre_adapt.data_loaders.red_pajama import RedPajamaPMD
from recpre_adapt.data_loaders.gsm8k import GSM8K
from recpre_adapt.data_loaders.math_pile import MathPilePMD
from recpre_adapt.data_loaders.testing import TestingDataLoaderWrapper
from recpre_adapt.raven_exit_model import *
from recpre_adapt.muon import Muon
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
    Returns a mask where 1 means don't attend and 0 means attend.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    mask = mask.to(device) if device is not None else mask
    return mask

def calculate_scores(
    model: RavenForCausalLM,
    latents: list[torch.Tensor],
    score_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> list[torch.Tensor]:
    with torch.no_grad():
        last_latent = latents[-1]
        target: torch.Tensor = torch.softmax(model.predict_from_latents(last_latent).logits, dim=-1) # type: ignore
        # target: torch.Tensor = torch.softmax(model.lm_head(last_latent).float(), dim=-1) # type: ignore
        scores = []
        for latent in latents:
            latent_prob: torch.Tensor = torch.softmax(model.predict_from_latents(latent).logits, dim=-1) # type: ignore
            # latent_prob: torch.Tensor = torch.softmax(model.lm_head(latent).float(), dim=-1) # type: ignore
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
    exit_model: Union[LatentTransformerExitModel, LTEExitModel, LatentDiffExitModel, LatentDiffEmbeddingExitModel],
    second_exit_model: Optional[LatentDiffExitModel], # the second exit model is just used to log gradients
    x: torch.Tensor,
    num_steps: int,
    min_loops: int,
    discount: float,
    is_eval: bool = False,
):
    assert min_loops >= 1

    penalty = 2 - discount

    with torch.no_grad():
        model.forward(x, attention_mask=None, num_steps=torch.tensor((num_steps,)))
        assert len(model.latents) == num_steps + 1
        latent_list: list[torch.Tensor] = model.latents
        input_embeds: torch.Tensor = model.input_embeds
        scores = calculate_scores(model, latent_list, score_cross_entropy)
        # last_layer_latents = []
        # for latent in latent_list:
        #     for block_idx, block in enumerate(model.transformer.coda, start=1):
        #         latent, _ = block(latent, model.freqs_cis[:, :x.shape[1]], -block_idx)
        #     latent = model.transformer.ln_f(latent)
        #     last_layer_latents.append(latent.clone())
        # scores = calculate_scores(model, last_layer_latents, score_cross_entropy)

    # loss for the last step. shape: batch_size, seq_len
    # expected_loss = scores[num_steps] * (penalty ** num_steps)
    expected_loss = torch.zeros_like(scores[num_steps])
    next_loss = scores[num_steps] * (penalty ** num_steps)
    expected_steps = torch.ones_like(scores[num_steps]) * num_steps
    expected_true_loss = scores[num_steps]

    if isinstance(exit_model, LTEExitModel) or isinstance(exit_model, LatentTransformerExitModel):
        attn_mask = generate_causal_mask(num_steps, model.device)
        # exclude the last latent, because we always exit on it
        latents = torch.stack(latent_list[:-1], dim=2)
        # Flatten latents from shape (batch_size, seq_len, recurrent_depth, n_embd) to (batch_size * seq_len, recurrent_depth, n_embd)
        latents = latents.flatten(start_dim=0, end_dim=1)
        if isinstance(exit_model, LTEExitModel):
            # flatten input_embeds from shape (batch_size, seq_len, n_embd) to (batch_size * seq_len, n_embd)
            input_embeds = input_embeds.flatten(start_dim=0, end_dim=1)
            entire_policy = exit_model.forward(input_embeds, latents, attn_mask=attn_mask)
        else:
            entire_policy = exit_model.forward(latents, attn_mask=attn_mask)
        # unflatten policy back to (batch_size, seq_len, recurrent_depth, 2)
        entire_policy = entire_policy.unflatten(dim=0, sizes=(x.shape[0], x.shape[1]))

    # Iterate backwards from numsteps - 1 to min_loops, since we need the next reward to calculate the current one
    for i in range(num_steps - 1, min_loops - 1, -1):
        # latent dimensions: batch_size, seq_len, n_embd
        # policy dimensions: batch_size, seq_len, 2
        if i == num_steps - 1 and second_exit_model is not None:
            # only use the second exit model for the last step to save those gradients
            policy = second_exit_model.forward(latent_list[i - 1], latent_list[i])
        elif isinstance(exit_model, LatentDiffExitModel):
            # policy = exit_model.forward(input_embeds, latent_list[i - 1], latent_list[i], attn_mask=attn_mask)
            policy = exit_model.forward(latent_list[i - 1], latent_list[i])
            # policy = exit_model.forward(last_layer_latents[i - 1], last_layer_latents[i])
        elif isinstance(exit_model, LatentDiffEmbeddingExitModel):
            policy = exit_model.forward(input_embeds, latent_list[i - 1], latent_list[i])
        else:
            policy = entire_policy[:, :, i, :]
        # 0 means we exit, 1 means we continue
        # NOTE: scores here are (0, inf), and we penalize an additional cost for each extra
        # expected_loss = policy[:, :, 0] * (scores[i]) * (penalty ** i) + policy[:, :, 1] * expected_loss
        loss = policy[:, :, 0] * (scores[i]) * (penalty ** i) + policy[:, :, 1] * next_loss
        expected_loss += loss
        with torch.no_grad():
            next_loss = loss.clone().detach()
        with torch.no_grad():
            expected_steps = policy[:, :, 0] * i + policy[:, :, 1] * expected_steps
            if is_eval:
                expected_true_loss = policy[:, :, 0] * scores[i] + policy[:, :, 1] * expected_true_loss

    expected_loss = expected_loss / (num_steps - min_loops)

    first_exit_loss = None
    first_exit_steps = None
    if is_eval:
        with torch.no_grad():
            first_exit_loss = torch.zeros_like(scores[num_steps])
            first_exit_steps = torch.zeros_like(scores[num_steps])
            exit_reached = torch.zeros_like(scores[num_steps]).bool()
            for i in range(min_loops, num_steps):
                if isinstance(exit_model, LatentDiffExitModel):
                    policy = exit_model.forward(latent_list[i - 1], latent_list[i])
                elif isinstance(exit_model, LatentDiffEmbeddingExitModel):
                    policy = exit_model.forward(input_embeds, latent_list[i - 1], latent_list[i])
                else:
                    policy = entire_policy[:, :, i, :]
                new_exits = policy[:, :, 0] > 0.5
                new_exits = new_exits & ~exit_reached
                first_exit_loss += torch.where(new_exits, scores[i], torch.zeros_like(scores[i]))
                first_exit_steps += torch.where(new_exits, torch.ones_like(scores[i]) * i, torch.zeros_like(scores[i]))
                exit_reached = exit_reached | new_exits
            remaining = ~exit_reached
            first_exit_loss += torch.where(remaining, scores[num_steps], torch.zeros_like(scores[num_steps]))
            first_exit_steps += torch.where(remaining, torch.ones_like(scores[num_steps]) * num_steps, torch.zeros_like(scores[num_steps]))
    # Backpropagate and update model
    return expected_loss, expected_steps, expected_true_loss, first_exit_loss, first_exit_steps

def get_lr_scheduler(optimizer, warmup_epochs, total_epochs, decay_type="cosine"):
    """
    Create a learning rate scheduler with warmup and decay.
    
    Args:
        optimizer: The optimizer to modify learning rates for
        warmup_epochs: Number of epochs for linear warmup
        total_epochs: Total number of training epochs
        decay_type: Type of decay after warmup ("cosine", "linear", or "step")
    
    Returns:
        A learning rate scheduler
    """
    def lr_lambda(epoch):
        # Linear warmup phase
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        
        # Decay phase
        if decay_type == "cosine":
            # Cosine decay from 1 to 0 over remaining epochs
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
        elif decay_type == "linear":
            # Linear decay from 1 to 0.1 over remaining epochs
            decay_ratio = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 1.0 - 0.9 * min(1, decay_ratio)
        elif decay_type == "step":
            # Step decay by 0.1 every 1/3 of remaining epochs
            decay_step = (total_epochs - warmup_epochs) / 3
            return 0.1 ** (int((epoch - warmup_epochs) / decay_step))
        else:
            # No decay
            return 1.0
            
    return LambdaLR(optimizer, lr_lambda)

def train(
    model: RavenForCausalLM,
    exit_model: Union[LatentTransformerExitModel, LTEExitModel, LatentDiffExitModel, LatentDiffEmbeddingExitModel],
    second_exit_model: Optional[LatentDiffExitModel], # the second exit model is just used to log & measure gradients
    pmd: PoorMansDataLoaderBase,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LambdaLR,
    num_steps: int,
    val_batchs: int = 10,
    training_epochs: int = 100,
    min_loops: int = 1,
    discount: float = 0.99,
    verbose: bool = True,
):
    model.eval()

    x_val = []
    for i in range(val_batchs):
        x, _ = pmd.get_batch("train")
        x_val.append(x)

    with torch.no_grad():
        scores_per_x = []
        for x in x_val:
            model.forward(x, attention_mask=None, num_steps=torch.tensor((num_steps,)))
            assert len(model.latents) == num_steps + 1
            latents: list[torch.Tensor] = model.latents
            scores = calculate_scores(model, latents, score_cross_entropy)
            # last_layer_latents = []
            # for latent in latents:
            #     for block_idx, block in enumerate(model.transformer.coda, start=1):
            #         latent, _ = block(latent, model.freqs_cis[:, :x.shape[1]], -block_idx)
            #     latent = model.transformer.ln_f(latent)
            #     last_layer_latents.append(latent.clone())
            # scores = calculate_scores(model, last_layer_latents, score_cross_entropy)
            scores_per_x.append(scores)
        for i in range(num_steps + 1):
            total = 0
            for x in range(len(x_val)):
                total += scores_per_x[x][i].mean().item()
            logging.info(f"Recurrence r={i}, Validation Score would be: {total / len(x_val)}")

    for epoch in range(training_epochs):
        x, _ = pmd.get_batch("train")

        # Train the exit model using the calculated rewards
        optimizer.zero_grad()

        expected_loss, expected_steps, _, _, _ = compute_loss(model, exit_model, second_exit_model, x, num_steps, min_loops, discount)
        avg_loss = torch.mean(expected_loss) * (10 ** 5)

        avg_loss.backward()
        if second_exit_model is not None and epoch % 10 == 0:
            for name, param in second_exit_model.named_parameters():
                logging.info(f"{name}: {param.grad}")

        optimizer.step()
        lr_scheduler.step()
        current_lr = lr_scheduler.get_last_lr()[0]

        if epoch < 30 or (verbose and epoch % 10 == 0):
            logging.info(f"Epoch {epoch}, Loss: {avg_loss.item()}, "\
                f"Expected Steps: {torch.mean(expected_steps).item()}"\
                f" std: {torch.std(expected_steps, dim=-1).data} "\
                f"max: {torch.max(expected_steps)}, "\
                f"LR: {current_lr:.6f}")

        if epoch % 10 == 0:
            with torch.no_grad():
                val_losses = []
                val_expected_steps = []
                val_true_losses = []
                val_first_exit_losses = []
                val_first_exit_steps = []
                for x in x_val:
                    expected_loss, expected_steps, true_loss, first_exit_loss, first_exit_steps = compute_loss(model, exit_model, second_exit_model, x, num_steps, min_loops, discount, is_eval=True)
                    val_losses.append(torch.mean(expected_loss))
                    val_expected_steps.append(expected_steps)
                    val_true_losses.append(true_loss)
                    val_first_exit_losses.append(first_exit_loss)
                    val_first_exit_steps.append(first_exit_steps)
                val_loss = torch.mean(torch.stack(val_losses))
                val_expected_steps = torch.stack(val_expected_steps)
                val_true_loss = torch.mean(torch.stack(val_true_losses))
                val_first_exit_loss = torch.mean(torch.stack(val_first_exit_losses))
                val_first_exit_steps = torch.stack(val_first_exit_steps)
                logging.info(f"Epoch {epoch}, True Loss: {val_true_loss.item()}, "\
                    f"First Exit Loss: {val_first_exit_loss.item()}, "\
                    f"Validation Loss: {val_loss.item()}, \n"\
                    f"Expected Steps: {torch.mean(val_expected_steps).item()}"\
                    f" std: {torch.std(val_expected_steps, dim=-1)}"\
                    f" max: {torch.max(val_expected_steps)}, min: {torch.min(val_expected_steps)}"\
                    f" First Exit Steps: {torch.mean(val_first_exit_steps).item()}"\
                    f" max: {torch.max(val_first_exit_steps)}, min: {torch.min(val_first_exit_steps)}"\
                    f" LR: {current_lr:.6f}")
        if epoch % 100 == 99:
            # Save the model
            torch.save(exit_model.state_dict(), f"exit_model_{epoch}.pt")


def main(
    dataset: Literal["red_pajama", "gsm8k", "math_pile"],
    optimizer_name: Literal["muon", "adamw"],
    num_steps: int = 32,
    batch_size: int = 2,
    seq_len: int = 1024,
    discount: float = 0.99,
    min_loops: int = 1,
    trial_mode: bool = False,
    training_epochs: int = 1000,
    learning_rate: float = 3e-5,
    warmup_epochs: int = 0,
    decay_type: Literal["cosine", "linear", "step", "none"] = "none",
):
    torch_seed = 42
    model: RavenForCausalLM = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")

    update_huggingface_implementation(model)

    model.to("cuda", dtype=torch.bfloat16) # type: ignore
    model.save_latents = True

    torch.manual_seed(torch_seed)

    if trial_mode:
        batch_size = 1

    if dataset == "red_pajama":
        pmd = RedPajamaPMD(model.device, tokenizer, batch_size, seq_len)
    elif dataset == "gsm8k":
        pmd = GSM8K(model.device, tokenizer, batch_size, seq_len)
    elif dataset == "math_pile":
        pmd = MathPilePMD(model.device, tokenizer, batch_size, seq_len)
    else:
        raise ValueError(f"Invalid dataset: {dataset}")
    if trial_mode:
        pmd = TestingDataLoaderWrapper(pmd)
    # exit_model = RavenExitModel(model.config)
    # exit_model = RavenBasicLatentExitModel(model.config)
    exit_model = RavenLatentExitModel(model.config)
    # exit_model = RavenLatentEmbeddingExitModel(model.config)
    # exit_model = RavenLatentTransformerExitModel(model.config)
    exit_model.to(dtype=torch.bfloat16, device=model.device)

    # This second model is used to guage the degree the gradient is vanishing
    # second_exit_model = RavenLatentExitModel(model.config)
    # second_exit_model.to(dtype=torch.bfloat16, device=model.device)

    if optimizer_name == "adamw":
        # optimizer = torch.optim.AdamW(list(exit_model.parameters()) + list(second_exit_model.parameters()), lr=learning_rate)
        optimizer = torch.optim.AdamW(list(exit_model.parameters()), lr=learning_rate)
    elif optimizer_name == "muon":
        muon_params = [p for p in exit_model.parameters() if p.ndim >= 2]
        adamw_params = [p for p in exit_model.parameters() if p.ndim < 2]
        optimizer = Muon(muon_params, adamw_params, lr=learning_rate * 30)
    else:
        raise ValueError(f"Invalid optimizer: {optimizer_name}")

    # Create learning rate scheduler
    lr_scheduler = get_lr_scheduler(optimizer, warmup_epochs, training_epochs, decay_type)

    params = {
            "dataset": dataset,
            "optimizer": optimizer_name,
            "num_steps": num_steps,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "training_epochs": training_epochs,
            "learning_rate": learning_rate,
            "min_loops": min_loops,
            "discount": discount,
            "trial_mode": trial_mode,
            "torch_seed": torch_seed,
            "model": exit_model.__class__.__name__,
            "model_variant": "prev+latent_transformer+10^5gradboost",
            "warmup_epochs": warmup_epochs,
            "decay_type": decay_type,
    }
    with open("training_params.json", "w") as f:
        json.dump(params, f, indent=4)
    logging.info(f"Training params: {params}")

    # Add a non-zero cost to encourage efficiency
    train(
        model,
        exit_model,
        None,
        pmd,
        optimizer,
        lr_scheduler,
        num_steps=num_steps, 
        training_epochs=training_epochs,
        val_batchs=1 if trial_mode else 10,
        min_loops=min_loops,
        discount=discount,
        verbose=False,
    )


if __name__ == "__main__":
    main(
        dataset="red_pajama",
        optimizer_name="adamw",
        batch_size=4,
        seq_len=512,
        num_steps=32,
        training_epochs=10000,
        discount=0.999,
        #  warmup_epochs=100,
        #  decay_type="cosine",
        min_loops=4,
        trial_mode=False,
    )
