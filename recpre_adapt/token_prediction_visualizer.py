import os
import sys
import torch
import gradio as gr
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# Add the parent directory to the Python path to make recpre importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recpre.raven_modeling_minimal import RavenForCausalLM
from recpre_adapt.train import update_huggingface_implementation

class TokenPredictionVisualizer:
    def __init__(self, model_name="tomg-group-umd/huginn-0125"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: RavenForCausalLM
        self.init_model()

        # cached states
        self.text: str = ""
        self.max_steps: int = 0
        self.tokens: list[int] = []
        self.token_strings: list[str] = []
        self.spans: list[tuple[int, int]] = []
        self.top_k: int = 10
        self.latents: list[torch.Tensor] = []
        self.logits: list[torch.Tensor] = []

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
            full_probs = torch.softmax(logits, dim=0)
            
            # Calculate probabilities for top-k tokens
            top_probs = full_probs[top_indices].cpu().numpy()

            top_tokens = [idx.item() for idx in top_indices]
            
            # Add the "other" category
            return top_tokens, top_probs
    
    def get_comparison(self, token_index, num_steps):
        ref_tokens, ref_probs = self.get_predictions(token_index, self.max_steps)
        test_tokens, test_probs = self.get_predictions(token_index, num_steps)
        return test_tokens, test_probs, ref_tokens, ref_probs

    def calculate_kl_divergence(self, reference_steps, test_steps):
        """Calculate KL divergence between reference and test distributions for all tokens."""
        kl_divergences = []
        with torch.no_grad():
            for token_idx in range(len(self.tokens)):
                # Get full distributions from both models (not just top-k)
                ref_logits = self.logits[reference_steps][0, token_idx, :]
                test_logits = self.logits[test_steps][0, token_idx, :]
                
                # Convert to probabilities
                ref_probs = torch.softmax(ref_logits, dim=0)
                test_probs = torch.softmax(test_logits, dim=0)
                
                # Calculate KL divergence: KL(P||Q) = Σ P(x) * log(P(x)/Q(x))
                # Adding small epsilon to avoid log(0)
                epsilon = 1e-10
                kl_div = torch.sum(ref_probs * torch.log((ref_probs + epsilon) / (test_probs + epsilon)))
                kl_divergences.append(kl_div.item())
        
        return kl_divergences
    
    def create_kl_visualization(self, kl_divergences):
        """Create HTML with KL divergence values displayed above each token."""
        html = ""
        for i, (start, end) in enumerate(self.spans):
            token_text = self.text[start:end]
            kl_value = kl_divergences[i]
            
            # Create a color gradient based on KL value (higher values = more red)
            max_kl = max(kl_divergences) if kl_divergences else 1.0
            normalized_kl = min(kl_value / max_kl, 1.0)  # Normalize to [0,1]
            r = min(255, int(255 * normalized_kl))
            g = min(255, int(255 * (1 - normalized_kl)))
            b = 0
            
            # Create HTML with KL value above each token
            html += f"""
            <div style="display:inline-block; text-align:center; margin:0 2px;">
              <div style="font-size:0.8em; color:rgb({r},{g},{b}); font-weight:bold;">{kl_value:.3f}</div>
              <span>{token_text}</span>
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

    def calculate_token_kl_by_steps(self, token_idx):
        """Calculate KL divergence for a single token across all step counts compared to max steps."""
        kl_values = []
        with torch.no_grad():
            # Get reference distribution from maximum steps
            ref_logits = self.logits[self.max_steps][0, token_idx, :]
            ref_probs = torch.softmax(ref_logits, dim=0)
            
            # Calculate KL divergence for each step
            for step in range(1, self.max_steps + 1):
                test_logits = self.logits[step][0, token_idx, :]
                test_probs = torch.softmax(test_logits, dim=0)
                
                # Calculate KL divergence: KL(P||Q) = Σ P(x) * log(P(x)/Q(x))
                epsilon = 1e-10
                kl_div = torch.sum(ref_probs * torch.log((ref_probs + epsilon) / (test_probs + epsilon)))
                kl_values.append(kl_div.item())
        
        return kl_values
    
    def plot_token_kl_progression(self, token_idx):
        """Create a plot showing KL divergence progression by step count for the selected token."""
        kl_values = self.calculate_token_kl_by_steps(token_idx)
        steps = list(range(1, self.max_steps + 1))
        
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(steps, kl_values, marker='o', linestyle='-', color='purple')
        ax.set_xlabel('Number of Steps')
        ax.set_ylabel('KL Divergence (vs. max steps)')
        ax.set_title(f'KL Divergence Progression for Token: "{self.token_strings[token_idx]}"')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set fixed y-axis limit to 2.0
        ax.set_ylim(0, 2.0)
        
        # Add a vertical line at the test_steps position
        if hasattr(self, 'current_test_steps'):
            ax.axvline(x=self.current_test_steps, color='red', linestyle='--', 
                       label=f'Current test steps: {self.current_test_steps}')
            ax.legend()
        
        return fig

    def visualize_predictions(self, char_index: int, reference_steps: int, test_steps: int):
        """Visualize token predictions for a selected character position."""
        # Find which token corresponds to the clicked position
        token_index = -1
        
        for i, (start, end) in enumerate(self.spans):
            if start <= char_index < end:
                token_index = i
                break
        
        if token_index == -1:
            return "No token found at this position.", None, None, None
        
        # Store current test steps for the KL progression plot
        self.current_test_steps = test_steps
        
        selected_token = self.token_strings[token_index]
        # Clean the token for display by removing special characters like Ġ
        display_token = selected_token.replace('Ġ', ' ')

        highlighted_text = f"{self.text[:self.spans[token_index][0]]}<span style='background-color: yellow'>{self.text[self.spans[token_index][0]:self.spans[token_index][1]]}</span>{self.text[self.spans[token_index][1]:]}"

        # Get predictions for both step counts
        test_tokens, test_probs, ref_tokens, ref_probs = self.get_comparison(token_index, test_steps)

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Reference steps plot
        self._plot_predictions(ax1, ref_tokens, ref_probs, f"Predictions with {reference_steps} steps", "skyblue")

        # Test steps plot
        self._plot_predictions(ax2, test_tokens, test_probs, f"Predictions with {test_steps} steps", "salmon")

        plt.tight_layout()
        
        # Generate KL progression plot
        kl_progression_plot = self.plot_token_kl_progression(token_index)

        # Generate token information
        token_info = f"Selected token: '{display_token}' (Index: {token_index})"

        return highlighted_text, fig, token_info, kl_progression_plot

    def visualize_kl_divergence(self, reference_steps: int, test_steps: int):
        """Create visualization of KL divergence for all tokens."""
        kl_divergences = self.calculate_kl_divergence(reference_steps, test_steps)
        return self.create_kl_visualization(kl_divergences)

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
    initial_kl_viz = visualizer.visualize_kl_divergence(default_ref_steps, default_test_steps)

    with gr.Blocks() as interface:
        gr.Markdown("# Token Prediction Visualizer")
        gr.Markdown("Click on any token in the text to see the model's predictions with different recurrence steps.")

        with gr.Row():
            with gr.Column(scale=1):
                top_k = gr.Number(minimum=1, maximum=100, value=10, step=1, label="Top K", interactive=True)
                reference_steps = gr.Number(minimum=1, maximum=64, value=default_ref_steps, step=1, label="Reference Steps", interactive=True)
                test_steps = gr.Slider(minimum=1, maximum=default_ref_steps, value=default_test_steps, step=1, label="Test Steps", interactive=True)

            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Input Text", 
                    lines=5,
                    value=default_text
                )

        # Add KL divergence visualization - initialize with calculated values
        kl_visualization = gr.HTML(label="KL Divergence Visualization", value=initial_kl_viz)
        token_info = gr.Textbox(label="Token Information")
        highlighted_text = gr.HTML(label="Highlighted Text")
        with gr.Row():
            with gr.Column(scale=3):
                prediction_plot = gr.Plot(label="Prediction Distributions")
            with gr.Column(scale=2):
                kl_progression_plot = gr.Plot(label="KL Divergence by Step Count")

        # Add a state to store the last clicked position
        last_click_position = gr.State(-1)

        # Handler for when text is clicked. NOTE: gradio passes in the evt based on type annotations
        def store_click_position(evt: gr.SelectData):
            return evt.index[0]

        # Handler for when sliders change
        def update_visualization(text: str, reference_steps: int, test_steps: int, char_index: int):
            if char_index == -1:  # No click has happened yet
                return highlighted_text.value, prediction_plot.value, token_info.value, kl_progression_plot.value
            assert reference_steps >= test_steps
            visualizer.process_text(text, max_steps=reference_steps)
            return visualizer.visualize_predictions(char_index, reference_steps, test_steps)

        # Update test_steps maximum when reference_steps changes
        def update_test_steps_max(reference_value):
            ref_value = int(reference_value)
            return gr.update(maximum=ref_value, value=min(int(test_steps.value), ref_value))

        # Handler for updating KL divergence visualization
        def update_kl_visualization(text: str, reference_steps: int, test_steps: int):
            visualizer.process_text(text, max_steps=reference_steps)
            return visualizer.visualize_kl_divergence(reference_steps, test_steps)

        # Store the click position when text is clicked and then update visualization
        text_input.select(
            fn=store_click_position,
            outputs=[last_click_position],
        ).then(  # Chain the second callback to run after the first
            fn=update_visualization,
            inputs=[text_input, reference_steps, test_steps, last_click_position],
            outputs=[highlighted_text, prediction_plot, token_info, kl_progression_plot],
        )
        text_input.change(
            fn=update_kl_visualization,
            inputs=[text_input, reference_steps, test_steps],
            outputs=[kl_visualization],
        )

        # Update visualization when sliders change
        reference_steps.change(
            fn=update_test_steps_max,
            inputs=[reference_steps],
            outputs=[test_steps]
        ).then(
            fn=update_visualization,
            inputs=[text_input, reference_steps, test_steps, last_click_position],
            outputs=[highlighted_text, prediction_plot, token_info, kl_progression_plot]
        ).then(
            fn=update_kl_visualization,
            inputs=[text_input, reference_steps, test_steps],
            outputs=[kl_visualization],
        )
        
        test_steps.change(
            fn=update_visualization,
            inputs=[text_input, reference_steps, test_steps, last_click_position],
            outputs=[highlighted_text, prediction_plot, token_info, kl_progression_plot]
        ).then(
            fn=update_kl_visualization,
            inputs=[text_input, reference_steps, test_steps],
            outputs=[kl_visualization],
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
            outputs=[highlighted_text, prediction_plot, token_info, kl_progression_plot]
        )

    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()
