from enum import Enum
import os
import sys
import logging
import json
from typing import Callable, Literal
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

# Add the parent directory to the Python path to make recpre importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recpre.raven_modeling_minimal import RavenForCausalLM, CausalSelfAttention
from recpre_adapt.checkpoint import save_checkpoint
from recpre_adapt.data_loaders import PoorMansDataLoaderBase
from recpre_adapt.data_loaders.red_pajama import RedPajamaPMD
from recpre_adapt.data_loaders.gsm8k import GSM8K
from recpre_adapt.data_loaders.math_pile import MathPilePMD
from recpre_adapt.data_loaders.testing import TestingDataLoaderWrapper
from recpre_adapt.raven_exit_model import *
from recpre_adapt.features import compute_feature_vectors
from recpre_adapt.muon import Muon
from recpre_adapt.score import calculate_scores, get_score_func, calculate_optimal_scores, SCORE_FUNCS
from recpre_adapt.score import score_cross_entropy, count_matching_topk
from recpre_adapt.utils import update_huggingface_implementation, generate_causal_mask, get_lr_scheduler


class EvalMetrics(Enum):
    TRUE_LOSS = "true_loss"
    CROSS_ENTROPY = "cross_entropy"
    TOP_1_MATCHES = "top_1_matches"
    TOP_3_MATCHES = "top_3_matches"
    TOP_5_MATCHES = "top_5_matches"


def make_modulo_sum_matrix(m: int,
                           n: int,
                           dtype: torch.dtype = torch.float32,
                           device=None) -> torch.Tensor:
    """
    Returns P of shape (n, m) such that for any x in R^m,
        y = P @ x
    is in R^n and
        y[i] = sum_{j: j % n == i} x[j].
    """
    # 1) allocate
    P = torch.zeros((n, m), dtype=dtype, device=device)
    # 2) compute for each column j which row it belongs to
    cols = torch.arange(m, device=device)
    rows = cols % n
    # 3) set those positions to 1
    P[rows, cols] = 1.0
    return P


def compute_loss(
    model: RavenForCausalLM,
    exit_model: Union[LatentTransformerExitModel, LTEExitModel, LatentDiffExitModel, LatentDiffEmbeddingExitModel, LatentRecurrentExitModel, FeatureRecurrentExitModel, ProbsRecurrentExitModel, TransformerExitModel],
    second_exit_model: Optional[LatentDiffExitModel], # the second exit model is just used to log gradients
    loss_func: Literal["rl", "optimal_ce"],
    score_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    num_steps: int,
    min_loops: int,
    loops_per_exit_eval: int,
    penalty: float,
    use_final_latents: bool,
    is_eval: bool = False,
):
    assert min_loops >= 1
    assert min_loops % loops_per_exit_eval == 0
    assert num_steps % loops_per_exit_eval == 0

    penalty = penalty ** loops_per_exit_eval

    with torch.no_grad():
        model.forward(x, attention_mask=None, num_steps=torch.tensor((num_steps,)))
        assert len(model.latents) == num_steps + 1
        latent_list: list[torch.Tensor] = model.latents

        # adjust the latents, min_loops and num_steps to the scale of the loops_per_exit_eval
        min_loops = min_loops // loops_per_exit_eval
        num_steps = num_steps // loops_per_exit_eval
        latent_list = latent_list[::loops_per_exit_eval]
        if use_final_latents:
            last_layer_latents = []
            for latent in latent_list:
                for block_idx, block in enumerate(model.transformer.coda, start=1):
                    latent, _ = block(latent, model.freqs_cis[:, :x.shape[1]], -block_idx)
                latent = model.transformer.ln_f(latent)
                last_layer_latents.append(latent.clone())
            latent_list = last_layer_latents

        if use_final_latents:
            logits: list[torch.Tensor] = [model.lm_head(latent).float() for latent in latent_list] # type: ignore
        else:
            logits: list[torch.Tensor] = [model.predict_from_latents(latent).logits for latent in latent_list] # type: ignore

        # probs: list[torch.Tensor] = [torch.softmax(model.lm_head(latent).float(), dim=-1) for latent in latent_list] # type: ignore
        input_embeds: torch.Tensor = model.input_embeds

        if isinstance(exit_model, ProbsRecurrentExitModel):
            P = make_modulo_sum_matrix(logits[0].shape[2], exit_model.recurrent_dim, device=logits[0].device)
            probs = [F.softmax(logit, dim=-1) @ P.T for logit in logits]

        scores = calculate_scores(logits, score_func)
        # scores = calculate_scores(model, last_layer_latents, score_cross_entropy)
        if loss_func == "optimal_ce":
            optimal_scores, optimal_indices = calculate_optimal_scores(logits, score_func, penalty)
        eval_scores = {}
        if is_eval:
            eval_scores = {
                EvalMetrics.TRUE_LOSS: scores,
                EvalMetrics.CROSS_ENTROPY: calculate_scores(logits, score_cross_entropy),
                EvalMetrics.TOP_1_MATCHES: calculate_scores(logits, lambda target, predicted: count_matching_topk(target, predicted, 1)),
                EvalMetrics.TOP_3_MATCHES: calculate_scores(logits, lambda target, predicted: count_matching_topk(target, predicted, 3)/3.0),
                EvalMetrics.TOP_5_MATCHES: calculate_scores(logits, lambda target, predicted: count_matching_topk(target, predicted, 5)/5.0),
            }
        # Clean up intermediate tensors

        del logits
        # del latent_list, input_embeds
        # gc.collect()
        # torch.cuda.empty_cache()

    if isinstance(exit_model, ProbsRecurrentExitModel):
        predicted_policy = exit_model.forward(probs)
    else:
        predicted_policy = compute_policy(model, exit_model, second_exit_model, x, num_steps, min_loops, latent_list[:-1], input_embeds)

    if loss_func == "rl":
        expected_loss = compute_rl_loss(predicted_policy, scores, num_steps, min_loops, penalty)
    elif loss_func == "optimal_ce":
        expected_loss = compute_optimal_policy_loss(predicted_policy, optimal_indices, num_steps, min_loops)
    else:
        raise ValueError(f"Invalid loss function: {loss_func}")

    if is_eval:
        assert eval_scores is not None
        expected_steps, expected_true_losses, first_exit_steps, first_exit_losses = compute_metrics(predicted_policy, scores, eval_scores, num_steps, min_loops)
        return expected_loss, loops_per_exit_eval * expected_steps, expected_true_losses, loops_per_exit_eval * first_exit_steps, first_exit_losses
    else:
        return expected_loss, None, {}, None, {}


def compute_policy(
    model: RavenForCausalLM,
    exit_model: Union[LatentTransformerExitModel, LTEExitModel, LatentDiffExitModel, LatentDiffEmbeddingExitModel, LatentRecurrentExitModel, FeatureRecurrentExitModel, TransformerExitModel],
    second_exit_model: Optional[LatentDiffExitModel],
    x: torch.Tensor,
    num_steps: int,
    min_loops: int,
    latent_list: list[torch.Tensor],
    input_embeds: torch.Tensor,
):
    if isinstance(exit_model, LTEExitModel) or isinstance(exit_model, LatentTransformerExitModel):
        attn_mask = generate_causal_mask(num_steps, model.device)
        # exclude the last latent, because we always exit on it
        latents = torch.stack(latent_list, dim=2)
        # Flatten latents from shape (batch_size, seq_len, recurrent_depth, n_embd) to (batch_size * seq_len, recurrent_depth, n_embd)
        latents = latents.flatten(start_dim=0, end_dim=1)
        if isinstance(exit_model, LTEExitModel):
            # flatten input_embeds from shape (batch_size, seq_len, n_embd) to (batch_size * seq_len, n_embd)
            input_embeds = input_embeds.flatten(start_dim=0, end_dim=1)
            entire_policy = exit_model.forward(input_embeds, latents, attn_mask=attn_mask)
        else:
            entire_policy = exit_model.forward(latents, attn_mask=attn_mask)
        # unflatten policy back to (batch_size, seq_len, recurrent_depth, 2)
        predicted_policy = entire_policy.unflatten(dim=0, sizes=(x.shape[0], x.shape[1]))
        return predicted_policy
    elif isinstance(exit_model, LatentRecurrentExitModel):
        predicted_policy = exit_model.forward(torch.stack(latent_list, dim=2))
        return predicted_policy
    elif isinstance(exit_model, FeatureRecurrentExitModel):
        latents = torch.stack(latent_list, dim=2)
        feature_vectors = compute_feature_vectors(latents)
        predicted_policy = exit_model.forward(input_embeds, feature_vectors)
        return predicted_policy
    elif isinstance(exit_model, TransformerExitModel):
        latents = torch.stack(latent_list, dim=2)
        predicted_policy = exit_model.forward(latents)
        return predicted_policy

    predicted_policy = torch.zeros(x.shape[0], x.shape[1], num_steps, 2, device=x.device, dtype=torch.float32)
    predicted_policy[:, :, :min_loops, 1] = 1.0

    for i in range(min_loops, num_steps):
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

        predicted_policy[:, :, i, :] = policy.float()

    return predicted_policy


@torch.no_grad()
def compute_metrics(
    predicted_policy: torch.Tensor,
    scores: list[torch.Tensor],
    eval_scores: dict[EvalMetrics, list[torch.Tensor]],
    num_steps: int,
    min_loops: int,
):
    expected_steps = torch.ones_like(scores[num_steps]) * num_steps
    expected_true_losses = {metric: eval_scores[metric][num_steps] for metric in EvalMetrics}
    for i in range(num_steps - 1, min_loops - 1, -1):
        policy = predicted_policy[:, :, i, :]
        expected_steps = policy[:, :, 0] * i + policy[:, :, 1] * expected_steps
        for metric in EvalMetrics:
            expected_true_losses[metric] = policy[:, :, 0] * eval_scores[metric][i] + policy[:, :, 1] * expected_true_losses[metric]

    first_exit_losses = {metric: torch.zeros_like(eval_scores[metric][num_steps]) for metric in EvalMetrics}
    first_exit_steps = torch.zeros_like(scores[num_steps])
    exit_reached = torch.zeros_like(scores[num_steps]).bool()
    for i in range(min_loops, num_steps):
        policy = predicted_policy[:, :, i, :]
        new_exits = policy[:, :, 0] > 0.5
        new_exits = new_exits & ~exit_reached
        for metric in EvalMetrics:
            first_exit_losses[metric] += torch.where(new_exits, eval_scores[metric][i], torch.zeros_like(eval_scores[metric][i]))
        first_exit_steps += torch.where(new_exits, torch.ones_like(scores[i]) * i, torch.zeros_like(scores[i]))
        exit_reached = exit_reached | new_exits
    remaining = ~exit_reached
    for metric in EvalMetrics:
        first_exit_losses[metric] += torch.where(remaining, eval_scores[metric][num_steps], torch.zeros_like(eval_scores[metric][num_steps]))
    first_exit_steps += torch.where(remaining, torch.ones_like(scores[num_steps]) * num_steps, torch.zeros_like(scores[num_steps]))

    return expected_steps, expected_true_losses, first_exit_steps, first_exit_losses


def compute_rl_loss(
    predicted_policy: torch.Tensor,
    scores: list[torch.Tensor],
    num_steps: int,
    min_loops: int,
    penalty: float,
):
    # loss for the last step. shape: batch_size, seq_len
    # expected_loss = scores[num_steps] * (penalty ** num_steps)
    expected_loss = torch.zeros_like(scores[num_steps])
    next_loss = scores[num_steps] * (penalty ** num_steps)

    # Iterate backwards from numsteps - 1 to min_loops, since we need the next reward to calculate the current one
    for i in range(num_steps - 1, min_loops - 1, -1):
        policy = predicted_policy[:, :, i, :]
        # 0 means we exit, 1 means we continue
        # NOTE: scores here are (0, inf), and we penalize an additional cost for each extra
        # expected_loss = policy[:, :, 0] * (scores[i]) * (penalty ** i) + policy[:, :, 1] * expected_loss
        loss = policy[:, :, 0] * (scores[i]) * (penalty ** i) + policy[:, :, 1] * next_loss
        expected_loss += loss
        with torch.no_grad():
            next_loss = loss.clone().detach()
    expected_loss = expected_loss / (num_steps - min_loops)
    return expected_loss


def compute_recursive_rl_loss(
    predicted_policy: torch.Tensor,
    scores: list[torch.Tensor],
    num_steps: int,
    min_loops: int,
    penalty: float,
):
    # loss for the last step. shape: batch_size, seq_len
    expected_loss = scores[num_steps] * (penalty ** num_steps)

    # Iterate backwards from numsteps - 1 to min_loops, since we need the next reward to calculate the current one
    for i in range(num_steps - 1, min_loops - 1, -1):
        policy = predicted_policy[:, :, i, :]
        # 0 means we exit, 1 means we continue
        expected_loss = policy[:, :, 0] * (scores[i]) * (penalty ** i) + policy[:, :, 1] * expected_loss
    return expected_loss


def compute_optimal_policy_loss(
    predicted_policy: torch.Tensor,
    optimal_indices: torch.Tensor,
    num_steps: int,
    min_loops: int,
):
    """
    Computes the cross-entropy loss between the exit model's policy and the
    theoretically optimal policy derived from minimizing future discounted scores.
    """

    # 3. Construct Optimal Policy (Target for Cross-Entropy)
    optimal_policy = torch.zeros_like(predicted_policy)
    # optimal_indices[b, s] = k means optimal to exit at step k (or later if k < min_loops)
    # We should continue for steps i < k and exit for steps i >= k (within the min_loops..num_steps-1 range)
    step_indices = torch.arange(num_steps, device=predicted_policy.device).view(1, 1, -1) # Shape (1, 1, num_steps)
    optimal_indices_expanded = optimal_indices.unsqueeze(-1) # Shape (batch_size, seq_len, 1)

    # Mask for continuing (step < optimal exit step)
    continue_mask = (step_indices < optimal_indices_expanded) # Shape (batch_size, seq_len, num_steps)
    # Mask for exiting (step >= optimal exit step)
    exit_mask = ~continue_mask # Shape (batch_size, seq_len, num_steps)

    optimal_policy[:, :, :, 1] = continue_mask.float() # Set continue probability (index 1) to 1 where mask is True
    optimal_policy[:, :, :, 0] = exit_mask.float()   # Set exit probability (index 0) to 1 where mask is True

    # 4. Calculate Cross-Entropy Loss for relevant steps
    # Select the slices corresponding to steps where decisions are made
    predicted_policy_slice = predicted_policy[:, :, min_loops:num_steps, :]
    optimal_policy_slice = optimal_policy[:, :, min_loops:num_steps, :]

    # Avoid division by zero if no decision steps exist
    if num_steps <= min_loops:
        return torch.tensor(0.0, device=predicted_policy.device, requires_grad=True) # No loss if no decisions made

    # Add epsilon for numerical stability to prevent log(0)
    epsilon = 1e-9
    # Compute cross-entropy: - sum(target * log(prediction)) over the two classes [exit, continue]
    ce_loss_per_element = - (optimal_policy_slice * torch.log(predicted_policy_slice + epsilon)).sum(dim=-1)

    # Average the loss over batch, sequence length, and the decision steps
    avg_loss = ce_loss_per_element.mean()

    return avg_loss


def report_validation_reference_scores(model, x_val, score_func, num_steps, penalty):
    with torch.no_grad():
        # scores_per_x = []
        eval_scores_per_x = []
        optimal_scores_per_x = []
        optimal_indices_per_x = []
        for x in x_val:
            model.forward(x, attention_mask=None, num_steps=torch.tensor((num_steps,)))
            assert len(model.latents) == num_steps + 1
            latents: list[torch.Tensor] = model.latents
            probs: list[torch.Tensor] = [model.predict_from_latents(latent).logits for latent in latents] # type: ignore
            # last_layer_latents = []
            # for latent in latents:
            #     for block_idx, block in enumerate(model.transformer.coda, start=1):
            #         latent, _ = block(latent, model.freqs_cis[:, :x.shape[1]], -block_idx)
            #     latent = model.transformer.ln_f(latent)
            #     last_layer_latents.append(latent.clone())
            # scores = calculate_scores(model, last_layer_latents, score_cross_entropy)
            optimal_scores, optimal_indices = calculate_optimal_scores(probs, score_func, penalty)
            optimal_scores_per_x.append(torch.mean(optimal_scores))
            optimal_indices_per_x.append(torch.mean(optimal_indices.float()))
            true_loss = calculate_scores(probs, score_func)
            eval_scores = {
                EvalMetrics.TRUE_LOSS: true_loss,
                EvalMetrics.CROSS_ENTROPY: calculate_scores(probs, score_cross_entropy),
                EvalMetrics.TOP_1_MATCHES: calculate_scores(probs, lambda target, predicted: count_matching_topk(target, predicted, 1)),
                EvalMetrics.TOP_3_MATCHES: calculate_scores(probs, lambda target, predicted: count_matching_topk(target, predicted, 3)/3.0),
                EvalMetrics.TOP_5_MATCHES: calculate_scores(probs, lambda target, predicted: count_matching_topk(target, predicted, 5)/5.0),
            }
            eval_scores_per_x.append(eval_scores)
            del latents, probs
            del optimal_scores, optimal_indices

        logging.info(f"Validation scores for reference")
        logging.info(f"Optimal scores: {torch.mean(torch.stack(optimal_scores_per_x))}, Optimal steps: {torch.mean(torch.stack(optimal_indices_per_x))}")
        all_eval_scores = []
        for i in range(num_steps + 1):
            eval_scores = {}
            for metric in EvalMetrics:
                total = 0
                for x in range(len(x_val)):
                    total += eval_scores_per_x[x][metric][i].mean().item()
                eval_scores[metric] = total / len(x_val)
            all_eval_scores.append(eval_scores)
            # logging.info(f"Recurrence r={i}: {eval_scores_per_x[0][EvalMetrics.TRUE_LOSS][i][0,:]}")
        for i in range(num_steps + 1):
            eval_score_strs = {k.value: f"{v:.5f}" for k, v in all_eval_scores[i].items()}
            eval_score_strs["discounted_loss"] = f"{all_eval_scores[i][EvalMetrics.TRUE_LOSS] * (penalty ** i):.5f}"
            logging.info(f"Recurrence r={i}: {eval_score_strs}")


def train(
    model: RavenForCausalLM,
    exit_model: Union[LatentTransformerExitModel, LTEExitModel, LatentDiffExitModel, LatentDiffEmbeddingExitModel, LatentRecurrentExitModel, FeatureRecurrentExitModel, ProbsRecurrentExitModel, TransformerExitModel],
    second_exit_model: Optional[LatentDiffExitModel], # the second exit model is just used to log & measure gradients
    loss_func: Literal["rl", "optimal_ce"],
    score_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pmd: PoorMansDataLoaderBase,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LambdaLR,
    num_steps: int,
    val_batchs: int = 10,
    training_epochs: int = 100,
    train_steps_per_x: int = 1,
    min_loops: int = 1,
    loops_per_exit_eval: int = 1,
    discount: float = 0.99,
    use_final_latents: bool = False,
    gradient_boost: float = 10**5,
    verbose: bool = True,
):
    penalty = 2 - discount

    model.eval()

    x_val = []
    for i in range(val_batchs):
        x, _ = pmd.get_batch("train")
        x_val.append(x)

    report_validation_reference_scores(model, x_val, score_func, num_steps, penalty)

    for epoch in range(training_epochs):
        if epoch % train_steps_per_x == 0:
            x, _ = pmd.get_batch("train")

        # Train the exit model using the calculated rewards
        optimizer.zero_grad()
        expected_loss, _, _, _, _ = compute_loss(model, exit_model, second_exit_model, loss_func, score_func, x, num_steps, min_loops, loops_per_exit_eval, penalty, use_final_latents)
        avg_loss = torch.mean(expected_loss) * gradient_boost

        avg_loss.backward()
        if second_exit_model is not None and epoch % 100 == 0:
            for name, param in second_exit_model.named_parameters():
                logging.info(f"{name}: {param.grad}")

        optimizer.step()
        lr_scheduler.step()
        current_lr = lr_scheduler.get_last_lr()[0]

        if epoch < 30 or (verbose and epoch % 10 == 0):
            logging.info(f"Epoch {epoch}, Loss: {avg_loss.item()}")

        if epoch % 10 == 9:
            with torch.no_grad():
                val_losses = []
                val_expected_steps = []
                val_true_losses = {metric: [] for metric in EvalMetrics}
                val_first_exit_losses = {metric: [] for metric in EvalMetrics}
                val_first_exit_steps = []
                for x in x_val:
                    expected_loss, expected_steps, true_losses, first_exit_steps, first_exit_losses = compute_loss(model, exit_model, second_exit_model, loss_func, score_func, x, num_steps, min_loops, loops_per_exit_eval, penalty, use_final_latents, is_eval=True)
                    val_losses.append(expected_loss)
                    val_expected_steps.append(expected_steps)
                    for metric in EvalMetrics:
                        val_true_losses[metric].append(true_losses[metric])
                        val_first_exit_losses[metric].append(first_exit_losses[metric])
                    val_first_exit_steps.append(first_exit_steps)
                val_loss = torch.mean(torch.stack(val_losses))
                val_expected_steps = torch.stack(val_expected_steps)
                val_first_exit_steps = torch.stack(val_first_exit_steps)
                val_true_loss = {}
                val_first_exit_loss = {}
                for metric in EvalMetrics:
                    val_true_loss[metric] = torch.mean(torch.stack(val_true_losses[metric]))
                    val_first_exit_loss[metric] = torch.mean(torch.stack(val_first_exit_losses[metric]))
                val_true_loss = {k.value: f"{v.item():.5f}" for k, v in val_true_loss.items()}
                val_first_exit_loss = {k.value: f"{v.item():.5f}" for k, v in val_first_exit_loss.items()}
                logging.info(f"Epoch {epoch}, True Loss: {val_true_loss}, "\
                    f"First Exit Loss: {val_first_exit_loss}, "\
                    f"Validation Loss: {val_loss}, \n"\
                    f"Expected Steps: {torch.mean(val_expected_steps).item()}"\
                    f" std: {torch.std(val_expected_steps, dim=-1)}"\
                    f" max: {torch.max(val_expected_steps)}, min: {torch.min(val_expected_steps)}"\
                    f" First Exit Steps: {torch.mean(val_first_exit_steps).item()}"\
                    f" max: {torch.max(val_first_exit_steps)}, min: {torch.min(val_first_exit_steps)}"\
                    f" LR: {current_lr:.6f}")
                # if epoch % 100 == 99:
                #     logging.info(f"{val_losses[0][0,:]}\n{val_expected_steps[0][0,:]}")
        if epoch % 500 == 499:
            # Save the model
            save_checkpoint(exit_model, optimizer, lr_scheduler, epoch, "exit_model")


def main(
    dataset: Literal["red_pajama", "gsm8k", "math_pile"],
    optimizer_name: Literal["muon", "adamw"],
    loss_func: Literal["rl", "optimal_ce"],
    score_func_name: SCORE_FUNCS,
    num_steps: int = 32,
    batch_size: int = 2,
    seq_len: int = 1024,
    discount: float = 0.99,
    use_final_latents: bool = False,
    min_loops: int = 1,
    loops_per_exit_eval: int = 1,
    trial_mode: bool = False,
    training_epochs: int = 1000,
    train_steps_per_x: int = 1,
    gradient_boost: float = 10**5,
    learning_rate: float = 3e-5,
    warmup_epochs: int = 0,
    decay_type: Literal["cosine", "linear", "step", "none"] = "none",
):
    # Set up logging
    log_file = "training.log" if not trial_mode else "training_trial.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    torch_seed = 42
    model: RavenForCausalLM = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", trust_remote_code=True, code_revision="2a364bd96e3eaa831be324f7c1f9e74892e4e594")
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")

    update_huggingface_implementation(model)

    model.to("cuda", dtype=torch.bfloat16) # type: ignore
    model.save_latents = True

    torch.manual_seed(torch_seed)
    
    autoencoder_path = None

    if trial_mode:
        batch_size = 1
        train_steps_per_x = 1

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
    # exit_model = RavenLatentExitModel(model.config)
    # exit_model = RavenLatentEmbeddingExitModel(model.config)
    # exit_model = RavenLatentTransformerExitModel(model.config)
    exit_model = RavenLatentTransformerExitModel2(model.config)
    # exit_model = RavenLatentRecurrentExitModel(model.config)
    # exit_model = RavenFeatureRecurrentExitModel(model.config, feature_count=5)
    # autoencoder_path = "checkpoints/ae_1/autoencoder_7000.pt"
    # autoencoder_path = "checkpoints/sae_1/autoencoder_4999.pt"
    # autoencoder_path = "checkpoints/sae_3/autoencoder_19999.pt"
    # autoencoder_path = "checkpoints/ae_2/autoencoder_9499.pt"
    # autoencoder = Autoencoder.load_from_checkpoint(autoencoder_path)
    # autoencoder.eval()

    # autoencoder.to(device=model.device)
    # print("done loading autoencoder to device")
    # exit_model = RavenAutoencoderRecurrentExitModel(autoencoder)
    # exit_model = RavenAutoencoderLTExitModel(autoencoder)
    # exit_model = RavenProbsRecurrentExitModel(model.config)
    # exit_model = RavenTransformerExitModel(model.config)
    # exit_model.seq_tf.load_state_dict(model.transformer.coda[0].state_dict())
    print("done creating exit model")
    exit_model.to(device=model.device)
    if not isinstance(exit_model, ProbsRecurrentExitModel):
        exit_model.to(dtype=torch.bfloat16)

    # This second model is used to guage the degree the gradient is vanishing
    # if trial_mode:
    #     second_exit_model = RavenLatentExitModel(model.config)
    #     second_exit_model.to(dtype=torch.bfloat16, device=model.device)
    # else:
    second_exit_model = None

    if optimizer_name == "adamw":
        # optimizer = torch.optim.AdamW(list(exit_model.parameters()) + list(second_exit_model.parameters()), lr=learning_rate)
        optimizer = torch.optim.AdamW(list(exit_model.parameters()), lr=learning_rate)
    elif optimizer_name == "muon":
        muon_params = [p for p in exit_model.parameters() if p.ndim >= 2]
        adamw_params = [p for p in exit_model.parameters() if p.ndim < 2]
        optimizer = Muon(muon_params, adamw_params, lr=learning_rate * 30)
    else:
        raise ValueError(f"Invalid optimizer: {optimizer_name}")

    score_func = get_score_func(score_func_name)


    # Create learning rate scheduler
    lr_scheduler = get_lr_scheduler(optimizer, warmup_epochs, training_epochs, decay_type)

    params = {
            "dataset": dataset,
            "optimizer": optimizer_name,
            "loss_func": loss_func,
            "score_func": score_func_name,
            "autoencoder_path": autoencoder_path,
            "num_steps": num_steps,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "training_epochs": training_epochs,
            "train_steps_per_x": train_steps_per_x,
            "learning_rate": learning_rate,
            "min_loops": min_loops,
            "loops_per_exit_eval": loops_per_exit_eval,
            "use_final_latents": use_final_latents,
            "discount": discount,
            "gradient_boost": gradient_boost,
            "trial_mode": trial_mode,
            "torch_seed": torch_seed,
            "model": exit_model.__class__.__name__,
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
        second_exit_model,
        loss_func,
        score_func,
        pmd,
        optimizer,
        lr_scheduler,
        num_steps=num_steps,
        training_epochs=training_epochs,
        train_steps_per_x=train_steps_per_x,
        val_batchs=1 if trial_mode else 10,
        min_loops=min_loops,
        loops_per_exit_eval=loops_per_exit_eval,
        discount=discount,
        use_final_latents=use_final_latents,
        gradient_boost=gradient_boost,
        verbose=False,
    )


if __name__ == "__main__":
    main(
        dataset="red_pajama",
        optimizer_name="adamw",
        loss_func="rl",
        score_func_name="ce",
        batch_size=2,
        seq_len=256,
        num_steps=24,
        training_epochs=10000,
        train_steps_per_x=2,
        discount=0.999,
        use_final_latents=False,
        warmup_epochs=100,
        decay_type="cosine",
        min_loops=8,
        loops_per_exit_eval=1,
        gradient_boost=1,
        trial_mode=False,
    )
