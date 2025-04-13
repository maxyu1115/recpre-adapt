from dataclasses import dataclass
import os
import sys
import json
from typing import Callable, Literal
import torch
import gradio as gr
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import argparse

# Add the parent directory to the Python path to make recpre importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recpre.raven_modeling_minimal import RavenForCausalLM
from recpre_adapt.train import update_huggingface_implementation, generate_causal_mask
from recpre_adapt.raven_exit_model import *

EPSILON = 1e-10

def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # Calculate KL divergence: KL(P||Q) = Σ P(x) * log(P(x)/Q(x))
    # Adding small epsilon to avoid log(0)
    return torch.sum(p * torch.log((p + EPSILON) / (q + EPSILON)), dim=-1)

def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))

def top_3_exact_match(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    top_3_p = torch.topk(p, 3, dim=-1).indices
    top_3_q = torch.topk(q, 3, dim=-1).indices
    return 3 - (torch.sum(top_3_p == top_3_q, dim=-1))

def top_5_exact_match(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    top_5_p = torch.topk(p, 5, dim=-1).indices
    top_5_q = torch.topk(q, 5, dim=-1).indices
    return 5 - (torch.sum(top_5_p == top_5_q, dim=-1))

def top_10_exact_match(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    top_10_p = torch.topk(p, 10, dim=-1).indices
    top_10_q = torch.topk(q, 10, dim=-1).indices
    return 10 - (torch.sum(top_10_p == top_10_q, dim=-1))


SCORE_FUNCTIONS = {
    "KL Divergence": (kl_divergence, 2.0),
    "JS Divergence": (js_divergence, 0.5),
    "Top 3 Exact Match": (top_3_exact_match, 4),
    "Top 5 Exact Match": (top_5_exact_match, 6),
    "Top 10 Exact Match": (top_10_exact_match, 11),
}


def latent_l1_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(p - q), dim=-1)

def latent_l2_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum(torch.pow(p - q, 2), dim=-1))

def entropy_diff(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return torch.sum(p * torch.log(p + EPSILON) - q * torch.log(q + EPSILON), dim=-1)


@dataclass
class ExitHeuristic:
    function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    limits: tuple[float, float]
    input_type: Literal["latent", "probs", "incremental-latent", "incremental-probs"]


EXIT_HEURISTIC_FUNCTIONS = {
    "Latent L1 Distance": ExitHeuristic(latent_l1_distance, (0.0, 10000.0), "latent"),
    "Latent L2 Distance": ExitHeuristic(latent_l2_distance, (0.0, 100.0), "latent"),
    "Entropy Diff": ExitHeuristic(entropy_diff, (-1.0, 1.0), "probs"),
    "Incremental Latent L1 Distance": ExitHeuristic(latent_l1_distance, (0.0, 10000.0), "incremental-latent"),
    "Incremental Latent L2 Distance": ExitHeuristic(latent_l2_distance, (0.0, 100.0), "incremental-latent"),
    "Incremental Entropy Diff": ExitHeuristic(entropy_diff, (-1.0, 1.0), "incremental-probs"),
}


class TokenPredictionVisualizer:
    def __init__(self, model_name="tomg-group-umd/huginn-0125", exit_model_dir="checkpoints/checkpoints_18/", exit_model_name="exit_model_2999.pt"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: RavenForCausalLM
        self.exit_model: Union[LatentTransformerExitModel, LTEExitModel, LatentDiffExitModel, LatentDiffEmbeddingExitModel]
        self.exit_model_min_steps: int
        self.exit_model_first_exit: bool = False
        self.init_model(exit_model_dir, exit_model_name)

        self.selected_score_function: str = "KL Divergence"
        self.score_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = SCORE_FUNCTIONS[self.selected_score_function][0]
        self.score_limit: float = SCORE_FUNCTIONS[self.selected_score_function][1]
        
        self.selected_exit_heuristic: str = "Latent L2 Distance"
        self.exit_heuristic: ExitHeuristic = EXIT_HEURISTIC_FUNCTIONS["Latent L2 Distance"]

        # cached states
        self.text: str = ""
        self.max_steps: int = 0
        self.tokens: list[int] = []
        self.token_strings: list[str] = []
        self.spans: list[tuple[int, int]] = []
        self.top_k: int = 10
        self.latents: list[torch.Tensor] = []
        self.input_embeds: torch.Tensor
        self.logits: list[torch.Tensor] = []
        self.probs: list[torch.Tensor] = []
        self.exit_policy: torch.Tensor
        self.expected_scores: torch.Tensor
        self.expected_probs: torch.Tensor
        self.expected_steps: torch.Tensor
        self.current_test_steps: int = 0

    def init_model(self, exit_model_dir: str, exit_model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
        update_huggingface_implementation(self.model)
        self.model.to(self.device, dtype=torch.bfloat16) # type: ignore
        self.model.save_latents = True
        self.model.eval()

        with open(os.path.join(exit_model_dir, "training_params.json"), "r") as f:
            training_params = json.load(f)

        if training_params.get("model") == "RavenLatentExitModel":
            self.exit_model = RavenLatentExitModel(self.model.config)
        elif training_params.get("model") == "RavenLatentEmbeddingExitModel":
            self.exit_model = RavenLatentEmbeddingExitModel(self.model.config)
        elif training_params.get("model") == "RavenLatentTransformerExitModel":
            self.exit_model = RavenLatentTransformerExitModel(self.model.config)
        else:
            self.exit_model = RavenExitModel(self.model.config)
        self.exit_model_min_steps = training_params.get("min_loops", 4)
        self.exit_model.load_state_dict(torch.load(os.path.join(exit_model_dir, exit_model_name)))
        self.exit_model.to(self.device, dtype=torch.bfloat16) # type: ignore
        self.exit_model.eval()

    def tokenize_text(self, text: str):
        """Tokenize text and return token strings and their indices."""
        self.tokens = self.tokenizer.encode(text, add_special_tokens=False)
        self.token_strings = self.tokenizer.convert_ids_to_tokens(self.tokens) # type: ignore
        encoding = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        self.spans = encoding.offset_mapping

    def get_token_index(self, char_index: int) -> int:
        if char_index < 0:
            return -1
        for i, (start, end) in enumerate(self.spans):
            if start <= char_index < end:
                return i
        return -1

    def process_text(self, text: str, max_steps: int, force_update: bool = False):
        if text == self.text and max_steps == self.max_steps and not force_update:
            return

        self.text = text
        self.tokenize_text(text)
        self.latents = []
        self.logits = []
        self.probs = []
        input_ids = torch.tensor([self.tokens]).to(self.device)
        
        with torch.no_grad():
            # Run the model with the specified number of steps
            self.model.forward(input_ids, attention_mask=None, num_steps=torch.tensor((max_steps,)))
            self.latents = self.model.latents
            self.input_embeds = self.model.input_embeds
            for i in range(max_steps + 1):
                logits = self.model.predict_from_latents(self.latents[i]).logits
                assert logits is not None
                self.logits.append(logits)
                probs = torch.softmax(logits, dim=-1)
                self.probs.append(probs)

            if isinstance(self.exit_model, LatentDiffExitModel):
                self.exit_policy = torch.zeros((input_ids.shape[0], input_ids.shape[1], max_steps, 2), device=self.device)
                for i in range(self.exit_model_min_steps, max_steps):
                    policy = self.exit_model.forward(self.latents[i - 1], self.latents[i])
                    self.exit_policy[:, :, i, :] = policy
            elif isinstance(self.exit_model, LTEExitModel) or isinstance(self.exit_model, LatentTransformerExitModel):
                latents = torch.stack(self.latents[:-1], dim=2)
                latents = latents.flatten(start_dim=0, end_dim=1)
                attn_mask = generate_causal_mask(max_steps, self.device)
                if isinstance(self.exit_model, LTEExitModel):
                    input_embeds = self.input_embeds.flatten(start_dim=0, end_dim=1)
                    policy = self.exit_model.forward(input_embeds, latents, attn_mask=attn_mask)
                elif isinstance(self.exit_model, LatentTransformerExitModel):
                    policy = self.exit_model.forward(latents, attn_mask=attn_mask)
                self.exit_policy = policy.unflatten(dim=0, sizes=(input_ids.shape[0], input_ids.shape[1]))
            else:
                raise ValueError(f"Not supported exit model type: {type(self.exit_model)}")
        self.max_steps = max_steps
        expected_score, expected_steps, expected_probs = self.calculate_exit_model_results()
        self.expected_steps = expected_steps.clone()
        self.expected_scores = expected_score.clone()
        self.expected_probs = expected_probs.clone()

    def get_predictions(self, probs: torch.Tensor, k: Optional[int] = None):
        with torch.no_grad():
            if k is None:
                k = self.top_k
            # Get the top k predictions
            top_probs, top_indices = torch.topk(probs, k)
            top_probs = top_probs.cpu().numpy()

            top_tokens = [idx.item() for idx in top_indices]

            # Add the "other" category
            return top_tokens, top_probs

    def set_score_function(self, function_name: str):
        """Set the score function to use for comparing distributions."""
        if function_name in SCORE_FUNCTIONS:
            self.selected_score_function = function_name
            self.score_function = SCORE_FUNCTIONS[function_name][0]
            self.score_limit = SCORE_FUNCTIONS[function_name][1]
            
    def set_exit_heuristic(self, heuristic_name: str):
        """Set the exit criteria function to use."""
        if heuristic_name in EXIT_HEURISTIC_FUNCTIONS:
            self.selected_exit_heuristic = heuristic_name
            self.exit_heuristic = EXIT_HEURISTIC_FUNCTIONS[heuristic_name]

    def calculate_scores(self, test_steps: int) -> torch.Tensor:
        """Calculate divergence scores between reference and test distributions for all tokens."""
        with torch.no_grad():
            # Get full distributions from both models (not just top-k)
            ref_logits = self.logits[self.max_steps][0, :, :]
            test_logits = self.logits[test_steps][0, :, :]

            # Convert to probabilities
            ref_probs = torch.softmax(ref_logits, dim=-1)
            test_probs = torch.softmax(test_logits, dim=-1)

            return self.score_function(ref_probs, test_probs)

    def calculate_top_k_matches(self, probs: torch.Tensor, ref_probs: torch.Tensor, top_k: int) -> int:
        top_tokens, top_probs = self.get_predictions(probs, top_k)
        top_ref_tokens, top_ref_probs = self.get_predictions(ref_probs, top_k)

        count = 0
        for i in range(top_k):
            count += top_tokens[i] == top_ref_tokens[i]
        return count

    def calculate_exit_model_results(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate divergence scores between reference and test distributions for all tokens."""
        with torch.no_grad():
            ref_logits = self.logits[self.max_steps][0, :, :]
            ref_probs = torch.softmax(ref_logits, dim=-1)
            ref_score = self.score_function(ref_probs, ref_probs)

            if not self.exit_model_first_exit:
                expected_score = ref_score
                expected_steps = torch.ones_like(expected_score) * self.max_steps
                expected_probs = ref_probs.clone()
                for i in range(self.max_steps - 1, self.exit_model_min_steps - 1, -1):
                    test_logits = self.logits[i][0, :, :]
                    test_probs = torch.softmax(test_logits, dim=-1)
                    score = self.score_function(ref_probs, test_probs)
                    expected_score = self.exit_policy[0, :, i, 0] * score + self.exit_policy[0, :, i, 1] * expected_score
                    if not self.exit_model_first_exit:
                        expected_steps = self.exit_policy[0, :, i, 0] * i + self.exit_policy[0, :, i, 1] * expected_steps
                        # print(self.exit_policy[0, :, i, 0].shape, test_probs.shape, expected_probs.shape)
                        expected_probs = self.exit_policy[0, :, i, 0].unsqueeze(-1) * test_probs + self.exit_policy[0, :, i, 1].unsqueeze(-1) * expected_probs
            else:
                expected_score = torch.zeros_like(ref_score)
                expected_steps = torch.zeros_like(ref_score)
                expected_probs = torch.zeros_like(ref_probs)
                exit_reached = torch.zeros_like(ref_score).bool()
                for i in range(self.exit_model_min_steps, self.max_steps):
                    test_logits = self.logits[i][0, :, :]
                    test_probs = torch.softmax(test_logits, dim=-1)
                    score = self.score_function(ref_probs, test_probs)
                    # policy = exit_model.forward(latent_list[i], latent_list[i + 1])
                    # new_exits = policy[:, :, 0] > 0.5
                    new_exits = self.exit_policy[0, :, i, 0] > 0.5
                    new_exits = new_exits & ~exit_reached
                    expected_steps += torch.where(new_exits, i, torch.zeros_like(expected_steps))
                    expected_probs += torch.where(new_exits.unsqueeze(-1), test_probs, torch.zeros_like(expected_probs))
                    expected_score += torch.where(new_exits, score, torch.zeros_like(expected_score))
                    exit_reached = exit_reached | new_exits
                remaining = ~exit_reached
                expected_steps += torch.where(remaining, self.max_steps, torch.zeros_like(expected_steps))
                expected_probs += torch.where(remaining.unsqueeze(-1), ref_probs, torch.zeros_like(expected_probs))
                expected_score += torch.where(remaining, ref_score, torch.zeros_like(expected_score))
            return expected_score, expected_steps, expected_probs

    def visualize_text_scores(self, test_steps: int, char_index: int) -> str:
        """
        Create HTML with score values displayed above each token, with highlighting for selected token.
        If selected_token_idx is < 0, no highlighting is done.
        """
        scores = self.calculate_scores(test_steps)
        selected_token_idx = self.get_token_index(char_index)
        html = ""
        ref_probs = torch.softmax(self.logits[self.max_steps][0, :, :], dim=-1)
        for i, (start, end) in enumerate(self.spans):
            token_text = self.text[start:end]
            score = scores[i].item()
            expected_score = self.expected_scores[i].item()
            expected_step = self.expected_steps[i].item()

            # Create a color gradient based on KL value (higher values = more red)
            max_score = scores.max().item()
            normalized_score = min(score / max_score, 1.0)  # Normalize to [0,1]
            r_score = min(255, int(255 * normalized_score))
            g_score = min(255, int(255 * (1 - normalized_score)))

            normalized_expected_score = min(expected_score / max_score, 1.0)
            r_expected_score = min(255, int(255 * normalized_expected_score))
            g_expected_score = min(255, int(255 * (1 - normalized_expected_score)))
            
            # normalized_expected_step = min((expected_step - self.exit_model_min_steps) / (self.max_steps - self.exit_model_min_steps), 1.0)
            if expected_step < test_steps:
                if test_steps < self.exit_model_min_steps:
                    normalized_expected_step = 0.0
                else:
                    normalized_expected_step = min((expected_step - self.exit_model_min_steps) / (test_steps - self.exit_model_min_steps), 1.0)
                r_expected_step = 0
                g_expected_step = min(255, int(255 * normalized_expected_step))
                b_expected_step = min(255, int(255 * (1 - normalized_expected_step)))
            else:
                normalized_expected_step = min((expected_step - test_steps) / (self.max_steps - test_steps), 1.0)
                r_expected_step = min(255, int(255 * normalized_expected_step))
                g_expected_step = min(255, int(255 * (1 - normalized_expected_step)))
                b_expected_step = 0
            
            k = 1
            top_k_matches = self.calculate_top_k_matches(self.expected_probs[i, :], ref_probs[i, :], k)

            # Add yellow background highlight for selected token
            if i == selected_token_idx:
                bg_style = "background-color: yellow;"
            elif top_k_matches < k:
                bg_style = "background-color: red;"
            else:
                bg_style = ""

            # Create HTML with score value above each token
            html += f"""
            <div style="display:inline-block; text-align:center; margin:0 2px;">
              <div style="font-size:0.8em; color:rgb({r_score},{g_score},0); font-weight:bold;">{score:.3f}</div>
              <div style="font-size:0.8em; font-weight:bold;">
                <span style="color:rgb({r_expected_score},{g_expected_score},0);">{expected_score:.3f}</span>
                <span style="color:rgb({r_expected_step},{g_expected_step},{b_expected_step});">({expected_step:.2f})</span>
              </div>
              <span style="{bg_style}">{token_text}</span>
            </div>
            """

        return f'<div style="line-height:1.6;">{html}</div>'

    def _plot_predictions(self, ax, tokens, probs, title, color):
        other = 1.0 - probs.sum()
        # Add labels for top-k tokens
        labels = [f"'{self.tokenizer.decode([token])}'[{token}] ({prob * 100:.1f}%)" for token, prob in zip(tokens, probs)]
        # Add the "other" category
        labels.append(f"Other ({other * 100:.1f}%)")
        # Combine top-k probabilities with "other" probability
        all_probs = np.append(probs, other)

        ax.barh(range(len(labels)), all_probs, color=[color]*len(probs) + ['lightgray'])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title(title)
        ax.set_xlabel("Probability")
        ax.invert_yaxis()

    def calculate_token_score_by_steps(self, token_idx) -> list[float]:
        """Calculate score for a single token across all step counts compared to max steps."""
        scores = []
        with torch.no_grad():
            # Get reference distribution from maximum steps
            ref_logits = self.logits[self.max_steps][0, token_idx, :]
            ref_probs = torch.softmax(ref_logits, dim=-1)

            # Calculate score for each step
            for step in range(1, self.max_steps + 1):
                test_logits = self.logits[step][0, token_idx, :]
                test_probs = torch.softmax(test_logits, dim=-1)

                score = self.score_function(ref_probs, test_probs)
                scores.append(score.item())
        return scores

    def calculate_token_exit_criteria_by_steps(self, token_idx) -> list[float]:
        """Calculate exit criteria for a single token across all step counts compared to max steps."""
        values = []
        with torch.no_grad():
            if self.exit_heuristic.input_type == "latent":
                # Get reference latent from maximum steps
                ref_latent = self.latents[self.max_steps][0, token_idx, :]

                # Calculate exit criteria for each step
                for step in range(1, self.max_steps + 1):
                    test_latent = self.latents[step][0, token_idx, :]
                    value = self.exit_heuristic.function(ref_latent, test_latent)
                    values.append(value.item())
            elif self.exit_heuristic.input_type == "probs":
                # Get reference probabilities from maximum steps
                ref_logits = self.logits[self.max_steps][0, token_idx, :]
                ref_probs = torch.softmax(ref_logits, dim=-1)

                # Calculate exit criteria for each step
                for step in range(1, self.max_steps + 1):
                    test_logits = self.logits[step][0, token_idx, :]
                    test_probs = torch.softmax(test_logits, dim=-1)
                    value = self.exit_heuristic.function(ref_probs, test_probs)
                    values.append(value.item())
            elif self.exit_heuristic.input_type == "incremental-latent":
                # Calculate exit criteria for each step
                for step in range(1, self.max_steps + 1):
                    prev_latent = self.latents[step - 1][0, token_idx, :]
                    test_latent = self.latents[step][0, token_idx, :]
                    value = self.exit_heuristic.function(prev_latent, test_latent)
                    values.append(value.item())
            elif self.exit_heuristic.input_type == "incremental-probs":
                # Calculate exit criteria for each step
                for step in range(1, self.max_steps + 1):
                    prev_logits = self.logits[step - 1][0, token_idx, :]
                    prev_probs = torch.softmax(prev_logits, dim=-1)
                    test_logits = self.logits[step][0, token_idx, :]
                    test_probs = torch.softmax(test_logits, dim=-1)
                    value = self.exit_heuristic.function(prev_probs, test_probs)
                    values.append(value.item())
        return values

    def plot_token_score_progression(self, token_idx):
        """Create a plot showing score progression by step count for the selected token."""
        scores = self.calculate_token_score_by_steps(token_idx)
        exit_criteria_values = self.calculate_token_exit_criteria_by_steps(token_idx)
        steps = list(range(1, self.max_steps + 1))

        fig, ax1 = plt.subplots(figsize=(10, 7))

        # Plot score on primary y-axis
        color1 = 'purple'
        ax1.set_xlabel('Number of Steps')
        ax1.set_ylabel(f'Score ({self.selected_score_function})', color=color1)
        ax1.plot(steps, scores, marker='o', linestyle='-', color=color1, label=self.selected_score_function)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim(0, self.score_limit)

        # Create secondary y-axis for exit criteria
        ax2 = ax1.twinx()
        color2 = 'green'
        ax2.set_ylabel(f'Exit Criteria ({self.selected_exit_heuristic})', color=color2)
        ax2.plot(steps, exit_criteria_values, marker='s', linestyle='--', color=color2, label=self.selected_exit_heuristic)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(self.exit_heuristic.limits[0], self.exit_heuristic.limits[1])
        
        ax1.set_title(f'Progression for Token: "{self.token_strings[token_idx]}"')
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Add a vertical line at the test_steps position
        if hasattr(self, 'current_test_steps'):
            ax1.axvline(x=self.current_test_steps, color='red', linestyle='--', 
                       label=f'Current test steps: {self.current_test_steps}')

        # Create a single legend for both curves
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        result_fig = fig
        plt.close(fig)
        return result_fig

    def visualize_selected_token(self, char_index: int, test_steps: int):
        """Visualize token predictions for a selected character position."""
        # Find which token corresponds to the clicked position
        token_index = self.get_token_index(char_index)

        if token_index == -1:
            return "No token found at this position.", None, None, None

        # Store current test steps for the KL progression plot
        self.current_test_steps = test_steps

        selected_token = self.token_strings[token_index]
        # Clean the token for display by removing special characters like Ġ
        display_token = selected_token.replace('Ġ', ' ')

        # Get predictions for both step counts
        test_tokens, test_probs = self.get_predictions(self.probs[test_steps][0, token_index, :])
        ref_tokens, ref_probs = self.get_predictions(self.probs[self.max_steps][0, token_index, :])
        exit_tokens, exit_probs = self.get_predictions(self.expected_probs[token_index, :])

        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Reference steps plot
        self._plot_predictions(ax1, ref_tokens, ref_probs, f"Predictions with {self.max_steps} steps", "skyblue")

        # Test steps plot
        self._plot_predictions(ax2, test_tokens, test_probs, f"Predictions with {test_steps} steps", "salmon")

        self._plot_predictions(ax3, exit_tokens, exit_probs, f"Early exit predictions with {self.expected_steps[token_index]:.3f} steps", "green")

        plt.tight_layout()

        # Generate score progression plot
        score_progression_plot = self.plot_token_score_progression(token_index)

        # Generate token information
        token_info = f"Selected token: '{display_token}' (Index: {token_index})"

        result_fig = fig
        plt.close(fig)
        return result_fig, token_info, score_progression_plot


def create_interface(visualizer: TokenPredictionVisualizer):
    """Create the Gradio interface."""
    # Default values
    default_text = """Question: Stella wanted to buy a new dress for the upcoming dance.  At the store she found out that the dress she wanted was $50.  
The store was offering 30% off of everything in the store.  What was the final cost of the dress?
Answer: The dress was $50 and 30% off so 50*.30 = $<<50*.30=15>>15 discount price
The dress cost $50 minus $15 (30% off discount) so 50-15 = $<<50-15=35>>35
#### 35
    """
    default_ref_steps = 32
    default_test_steps = 8

    # Initialize visualizer with default text and steps
    visualizer.process_text(default_text, max_steps=default_ref_steps)
    initial_score_viz = visualizer.visualize_text_scores(default_test_steps, -1)

    with gr.Blocks() as interface:
        gr.Markdown("# Token Prediction Visualizer")
        gr.Markdown("Click on any token in the text to see the model's predictions with different recurrence steps.")

        with gr.Row():
            with gr.Column(scale=1):
                top_k = gr.Number(minimum=1, maximum=100, value=10, step=1, label="Top K", interactive=True)
                score_function_dropdown = gr.Dropdown(
                    choices=list(SCORE_FUNCTIONS.keys()),
                    value=visualizer.selected_score_function,
                    label="Score Function",
                    interactive=True
                )
                exit_heuristic_dropdown = gr.Dropdown(
                    choices=list(EXIT_HEURISTIC_FUNCTIONS.keys()),
                    value=visualizer.selected_exit_heuristic,
                    label="Exit Heuristic",
                    interactive=True
                )

            with gr.Column(scale=1):
                reference_steps = gr.Number(minimum=1, maximum=64, value=default_ref_steps, step=1, label="Reference Steps", interactive=True)
                test_steps = gr.Slider(minimum=1, maximum=default_ref_steps, value=default_test_steps, step=1, label="Test Steps", interactive=True)
                exit_criteria_dropdown = gr.Dropdown(
                    choices=["Expected Policy", "First >50% Exit"],
                    label="Exit Criteria",
                    interactive=True
                )

            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Input Text", 
                    lines=5,
                    value=default_text
                )

        # Add score visualization - initialize with calculated values
        score_visualization = gr.HTML(label="Score Visualization", value=initial_score_viz)
        token_info = gr.Textbox(label="Token Information")
        with gr.Row():
            with gr.Column(scale=4):
                prediction_plot = gr.Plot(label="Prediction Distributions")
            with gr.Column(scale=2):
                score_progression_plot = gr.Plot(label="Score by Step Count")

        # Add a state to store the last clicked position
        last_click_position = gr.State(-1)

        # Handler for when text is clicked. NOTE: gradio passes in the evt based on type annotations
        def store_click_position(evt: gr.SelectData):
            return evt.index[0]

        # Handler for when sliders change
        def update_visualization(text: str, reference_steps: int, test_steps: int, char_index: int):
            if char_index == -1:  # No click has happened yet
                return prediction_plot.value, token_info.value, score_progression_plot.value
            assert reference_steps >= test_steps
            visualizer.process_text(text, max_steps=reference_steps)
            return visualizer.visualize_selected_token(char_index, test_steps)

        # Update test_steps maximum when reference_steps changes
        def update_test_steps_max(reference_value):
            ref_value = int(reference_value)
            return gr.update(maximum=ref_value, value=min(int(test_steps.value), ref_value))

        # Handler for updating score visualization
        def update_score_visualization(text: str, reference_steps: int, test_steps: int, char_index: int):
            visualizer.process_text(text, max_steps=reference_steps)
            return visualizer.visualize_text_scores(test_steps, char_index)

        # Handler for updating score function
        def update_score_function(function_name):
            visualizer.set_score_function(function_name)

        # Handler for updating exit criteria
        def update_exit_heuristic(heuristic_name):
            visualizer.set_exit_heuristic(heuristic_name)
            
        def update_exit_criteria(criteria_name):
            if criteria_name == "Expected Policy":
                visualizer.exit_model_first_exit = False
            else:
                visualizer.exit_model_first_exit = True
            visualizer.process_text(text_input.value, reference_steps.value, force_update=True)

        # Store the click position when text is clicked and then update visualization
        text_input.select(
            fn=store_click_position,
            outputs=[last_click_position],
        ).then(  # Chain the second callback to run after the first
            fn=update_visualization,
            inputs=[text_input, reference_steps, test_steps, last_click_position],
            outputs=[prediction_plot, token_info, score_progression_plot],
        ).then(
            fn=update_score_visualization,
            inputs=[text_input, reference_steps, test_steps, last_click_position],
            outputs=[score_visualization],
        )
        text_input.change(
            fn=update_score_visualization,
            inputs=[text_input, reference_steps, test_steps, last_click_position],
            outputs=[score_visualization],
        )

        # Update visualization when sliders change
        reference_steps.change(
            fn=update_test_steps_max,
            inputs=[reference_steps],
            outputs=[test_steps]
        ).then(
            fn=update_visualization,
            inputs=[text_input, reference_steps, test_steps, last_click_position],
            outputs=[prediction_plot, token_info, score_progression_plot]
        ).then(
            fn=update_score_visualization,
            inputs=[text_input, reference_steps, test_steps, last_click_position],
            outputs=[score_visualization],
        )
        
        test_steps.change(
            fn=update_visualization,
            inputs=[text_input, reference_steps, test_steps, last_click_position],
            outputs=[prediction_plot, token_info, score_progression_plot]
        ).then(
            fn=update_score_visualization,
            inputs=[text_input, reference_steps, test_steps, last_click_position],
            outputs=[score_visualization],
        )

        # Update visualization and score visualization when score function changes
        score_function_dropdown.change(
            fn=update_score_function,
            inputs=[score_function_dropdown],
            outputs=[]
        ).then(
            fn=update_visualization,
            inputs=[text_input, reference_steps, test_steps, last_click_position],
            outputs=[prediction_plot, token_info, score_progression_plot]
        ).then(
            fn=update_score_visualization,
            inputs=[text_input, reference_steps, test_steps, last_click_position],
            outputs=[score_visualization],
        )

        # Update visualization when exit criteria changes
        exit_heuristic_dropdown.change(
            fn=update_exit_heuristic,
            inputs=[exit_heuristic_dropdown],
            outputs=[]
        ).then(
            fn=update_visualization,
            inputs=[text_input, reference_steps, test_steps, last_click_position],
            outputs=[prediction_plot, token_info, score_progression_plot]
        )

        exit_criteria_dropdown.change(
            fn=update_exit_criteria,
            inputs=[exit_criteria_dropdown],
            outputs=[]
        ).then(
            fn=update_visualization,
            inputs=[text_input, reference_steps, test_steps, last_click_position],
            outputs=[prediction_plot, token_info, score_progression_plot]
        ).then(
            fn=update_score_visualization,
            inputs=[text_input, reference_steps, test_steps, last_click_position],
            outputs=[score_visualization],
        )

        def update_top_k(x):
            visualizer.top_k = int(x)
            # return gr.update(value=int(x))

        top_k.change(
            fn=update_top_k,
            inputs=[top_k],
            outputs=[]
        ).then(
            fn=update_visualization,
            inputs=[text_input, reference_steps, test_steps, last_click_position],
            outputs=[prediction_plot, token_info, score_progression_plot]
        )

    return interface

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Token Prediction Visualizer")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio sharing."
    )
    parser.add_argument(
        "--exit-model-dir",
        type=str,
        default="checkpoints/checkpoints_18/",
        help="Path to the exit model checkpoint directory."
    )
    parser.add_argument(
        "--exit-model-name",
        type=str,
        default="exit_model_2999.pt",
        help="Name of the exit model checkpoint."
    )

    # Parse arguments
    args = parser.parse_args()

    # Create the visualizer instance with parsed arguments
    visualizer = TokenPredictionVisualizer(
        exit_model_dir=args.exit_model_dir,
        exit_model_name=args.exit_model_name
    )

    # Create and launch the interface
    interface = create_interface(visualizer)
    interface.launch(share=args.share)
