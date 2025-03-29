from dataclasses import dataclass
import os
import sys
from typing import Callable, Literal
import torch
import gradio as gr
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# Add the parent directory to the Python path to make recpre importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recpre.raven_modeling_minimal import RavenForCausalLM
from recpre_adapt.train import update_huggingface_implementation


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
class ExitCriteria:
    function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    limits: tuple[float, float]
    input_type: Literal["latent", "probs", "incremental-latent", "incremental-probs"]


EXIT_CRITERIA_FUNCTIONS = {
    "Latent L1 Distance": ExitCriteria(latent_l1_distance, (0.0, 10000.0), "latent"),
    "Latent L2 Distance": ExitCriteria(latent_l2_distance, (0.0, 100.0), "latent"),
    "Entropy Diff": ExitCriteria(entropy_diff, (-1.0, 1.0), "probs"),
    "Incremental Latent L1 Distance": ExitCriteria(latent_l1_distance, (0.0, 10000.0), "incremental-latent"),
    "Incremental Latent L2 Distance": ExitCriteria(latent_l2_distance, (0.0, 100.0), "incremental-latent"),
    "Incremental Entropy Diff": ExitCriteria(entropy_diff, (-1.0, 1.0), "incremental-probs"),
}


class TokenPredictionVisualizer:
    def __init__(self, model_name="tomg-group-umd/huginn-0125"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: RavenForCausalLM
        self.init_model()

        self.selected_score_function: str = "KL Divergence"
        self.score_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = SCORE_FUNCTIONS[self.selected_score_function][0]
        self.score_limit: float = SCORE_FUNCTIONS[self.selected_score_function][1]
        
        self.selected_exit_criteria: str = "Latent L2 Distance"
        self.exit_criteria: ExitCriteria = EXIT_CRITERIA_FUNCTIONS["Latent L2 Distance"]

        # cached states
        self.text: str = ""
        self.max_steps: int = 0
        self.tokens: list[int] = []
        self.token_strings: list[str] = []
        self.spans: list[tuple[int, int]] = []
        self.top_k: int = 10
        self.latents: list[torch.Tensor] = []
        self.logits: list[torch.Tensor] = []
        self.current_test_steps: int = 0

    def init_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
        update_huggingface_implementation(self.model)
        self.model.to(self.device, dtype=torch.bfloat16) # type: ignore
        self.model.save_latents = True
        self.model.eval()

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

    def process_text(self, text: str, max_steps: int):
        if text == self.text and max_steps == self.max_steps:
            return

        self.text = text
        self.tokenize_text(text)
        self.latents = []
        self.logits = []
        input_ids = torch.tensor([self.tokens]).to(self.device)
        
        with torch.no_grad():
            # Run the model with the specified number of steps
            self.model.forward(input_ids, attention_mask=None, num_steps=torch.tensor((max_steps,)))
            self.latents = self.model.latents
            for i in range(max_steps + 1):
                logits = self.model.predict_from_latents(self.latents[i]).logits
                assert logits is not None
                self.logits.append(logits)
        self.max_steps = max_steps

    def get_predictions(self, token_index, num_steps):
        with torch.no_grad():
            # Get the logits from the last latent state
            logits = self.logits[num_steps][0, token_index, :]
            
            # Get the top k predictions
            top_logits, top_indices = torch.topk(logits, self.top_k)
            
            # Calculate full softmax over all tokens
            full_probs = torch.softmax(logits, dim=-1)
            
            # Calculate probabilities for top-k tokens
            top_probs = full_probs[top_indices].cpu().numpy()

            top_tokens = [idx.item() for idx in top_indices]
            
            # Add the "other" category
            return top_tokens, top_probs
    
    def get_comparison(self, token_index, num_steps):
        ref_tokens, ref_probs = self.get_predictions(token_index, self.max_steps)
        test_tokens, test_probs = self.get_predictions(token_index, num_steps)
        return test_tokens, test_probs, ref_tokens, ref_probs

    def set_score_function(self, function_name: str):
        """Set the score function to use for comparing distributions."""
        if function_name in SCORE_FUNCTIONS:
            self.selected_score_function = function_name
            self.score_function = SCORE_FUNCTIONS[function_name][0]
            self.score_limit = SCORE_FUNCTIONS[function_name][1]
            
    def set_exit_criteria(self, criteria_name: str):
        """Set the exit criteria function to use."""
        if criteria_name in EXIT_CRITERIA_FUNCTIONS:
            self.selected_exit_criteria = criteria_name
            self.exit_criteria = EXIT_CRITERIA_FUNCTIONS[criteria_name]

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

    def create_score_visualization(self, scores: torch.Tensor, selected_token_idx: int = -1) -> str:
        """
        Create HTML with score values displayed above each token, with highlighting for selected token.
        If selected_token_idx is < 0, no highlighting is done.
        """
        html = ""
        for i, (start, end) in enumerate(self.spans):
            token_text = self.text[start:end]
            score = scores[i].item()

            # Create a color gradient based on KL value (higher values = more red)
            max_score = scores.max().item()
            normalized_score = min(score / max_score, 1.0)  # Normalize to [0,1]
            r = min(255, int(255 * normalized_score))
            g = min(255, int(255 * (1 - normalized_score)))
            b = 0

            # Add yellow background highlight for selected token
            bg_style = "background-color: yellow;" if i == selected_token_idx else ""

            # Create HTML with score value above each token
            html += f"""
            <div style="display:inline-block; text-align:center; margin:0 2px;">
              <div style="font-size:0.8em; color:rgb({r},{g},{b}); font-weight:bold;">{score:.3f}</div>
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
            if self.exit_criteria.input_type == "latent":
                # Get reference latent from maximum steps
                ref_latent = self.latents[self.max_steps][0, token_idx, :]

                # Calculate exit criteria for each step
                for step in range(1, self.max_steps + 1):
                    test_latent = self.latents[step][0, token_idx, :]
                    value = self.exit_criteria.function(ref_latent, test_latent)
                    values.append(value.item())
            elif self.exit_criteria.input_type == "probs":
                # Get reference probabilities from maximum steps
                ref_logits = self.logits[self.max_steps][0, token_idx, :]
                ref_probs = torch.softmax(ref_logits, dim=-1)

                # Calculate exit criteria for each step
                for step in range(1, self.max_steps + 1):
                    test_logits = self.logits[step][0, token_idx, :]
                    test_probs = torch.softmax(test_logits, dim=-1)
                    value = self.exit_criteria.function(ref_probs, test_probs)
                    values.append(value.item())
            elif self.exit_criteria.input_type == "incremental-latent":
                # Calculate exit criteria for each step
                for step in range(1, self.max_steps + 1):
                    prev_latent = self.latents[step - 1][0, token_idx, :]
                    test_latent = self.latents[step][0, token_idx, :]
                    value = self.exit_criteria.function(prev_latent, test_latent)
                    values.append(value.item())
            elif self.exit_criteria.input_type == "incremental-probs":
                # Calculate exit criteria for each step
                for step in range(1, self.max_steps + 1):
                    prev_logits = self.logits[step - 1][0, token_idx, :]
                    prev_probs = torch.softmax(prev_logits, dim=-1)
                    test_logits = self.logits[step][0, token_idx, :]
                    test_probs = torch.softmax(test_logits, dim=-1)
                    value = self.exit_criteria.function(prev_probs, test_probs)
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
        ax2.set_ylabel(f'Exit Criteria ({self.selected_exit_criteria})', color=color2)
        ax2.plot(steps, exit_criteria_values, marker='s', linestyle='--', color=color2, label=self.selected_exit_criteria)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(self.exit_criteria.limits[0], self.exit_criteria.limits[1])
        
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

    def visualize_predictions(self, char_index: int, reference_steps: int, test_steps: int):
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
        test_tokens, test_probs, ref_tokens, ref_probs = self.get_comparison(token_index, test_steps)

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Reference steps plot
        self._plot_predictions(ax1, ref_tokens, ref_probs, f"Predictions with {reference_steps} steps", "skyblue")

        # Test steps plot
        self._plot_predictions(ax2, test_tokens, test_probs, f"Predictions with {test_steps} steps", "salmon")

        plt.tight_layout()

        # Generate score progression plot
        score_progression_plot = self.plot_token_score_progression(token_index)

        # Generate token information
        token_info = f"Selected token: '{display_token}' (Index: {token_index})"

        result_fig = fig
        plt.close(fig)
        return result_fig, token_info, score_progression_plot

    def visualize_text_scores(self, test_steps: int, char_index: int):
        """Create visualization of score for all tokens with optional highlighting."""
        scores = self.calculate_scores(test_steps)
        return self.create_score_visualization(scores, self.get_token_index(char_index))


def create_interface():
    """Create the Gradio interface."""
    visualizer = TokenPredictionVisualizer()

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
                exit_criteria_dropdown = gr.Dropdown(
                    choices=list(EXIT_CRITERIA_FUNCTIONS.keys()),
                    value=visualizer.selected_exit_criteria,
                    label="Exit Criteria",
                    interactive=True
                )

            with gr.Column(scale=1):
                reference_steps = gr.Number(minimum=1, maximum=64, value=default_ref_steps, step=1, label="Reference Steps", interactive=True)
                test_steps = gr.Slider(minimum=1, maximum=default_ref_steps, value=default_test_steps, step=1, label="Test Steps", interactive=True)

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
            with gr.Column(scale=3):
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
            return visualizer.visualize_predictions(char_index, reference_steps, test_steps)

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
        def update_exit_criteria(criteria_name):
            visualizer.set_exit_criteria(criteria_name)

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
        exit_criteria_dropdown.change(
            fn=update_exit_criteria,
            inputs=[exit_criteria_dropdown],
            outputs=[]
        ).then(
            fn=update_visualization,
            inputs=[text_input, reference_steps, test_steps, last_click_position],
            outputs=[prediction_plot, token_info, score_progression_plot]
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
    interface = create_interface()
    interface.launch()
