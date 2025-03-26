import os
import sys
import torch
import gradio as gr
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

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

            # Get the top 10 predictions
            top_logits, top_indices = torch.topk(logits, 10)
            probs = torch.softmax(top_logits, dim=0).cpu().numpy()
            top_tokens = [self.tokenizer.decode([idx.item()]) for idx in top_indices]
        return top_tokens, probs
    
    def get_comparison(self, token_index, num_steps):
        ref_tokens, ref_probs = self.get_predictions(token_index, self.max_steps)
        test_tokens, test_probs = self.get_predictions(token_index, num_steps)
        return test_tokens, test_probs, ref_tokens, ref_probs

    def visualize_predictions(self, char_index: int, reference_steps: int, test_steps: int):
        """Visualize token predictions for a selected character position."""
        # Find which token corresponds to the clicked position
        token_index = -1
        
        for i, (start, end) in enumerate(self.spans):
            if start <= char_index < end:
                token_index = i
                break
        
        if token_index == -1:
            return "No token found at this position.", None, None
        
        selected_token = self.token_strings[token_index]
        # Clean the token for display by removing special characters like Ġ
        display_token = selected_token.replace('Ġ', ' ')

        highlighted_text = f"{self.text[:self.spans[token_index][0]]}<span style='background-color: yellow'>{self.text[self.spans[token_index][0]:self.spans[token_index][1]]}</span>{self.text[self.spans[token_index][1]:]}"

        # Get predictions for both step counts
        test_tokens, test_probs, ref_tokens, ref_probs = self.get_comparison(token_index, test_steps)

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Reference steps plot
        ax1.barh(range(len(ref_tokens)), ref_probs, color='skyblue')
        ax1.set_yticks(range(len(ref_tokens)))
        ax1.set_yticklabels([f"{token} ({prob:.3f})" for token, prob in zip(ref_tokens, ref_probs)])
        ax1.set_title(f"Predictions with {reference_steps} steps")
        ax1.set_xlabel("Probability")
        ax1.invert_yaxis()

        # Test steps plot
        ax2.barh(range(len(test_tokens)), test_probs, color='salmon')
        ax2.set_yticks(range(len(test_tokens)))
        ax2.set_yticklabels([f"{token} ({prob:.3f})" for token, prob in zip(test_tokens, test_probs)])
        ax2.set_title(f"Predictions with {test_steps} steps")
        ax2.set_xlabel("Probability")
        ax2.invert_yaxis()

        plt.tight_layout()

        # Generate token information
        token_info = f"Selected token: '{display_token}' (Index: {token_index})"

        return highlighted_text, fig, token_info

def create_interface():
    """Create the Gradio interface."""
    visualizer = TokenPredictionVisualizer()

    with gr.Blocks() as interface:
        gr.Markdown("# Token Prediction Visualizer")
        gr.Markdown("Click on any token in the text to see the model's predictions with different recurrence steps.")

        with gr.Row():
            with gr.Column(scale=1):
                reference_steps = gr.Number(minimum=1, maximum=32, value=16, step=1, label="Reference Steps", interactive=True)
                test_steps = gr.Slider(minimum=1, maximum=16, value=8, step=1, label="Test Steps", interactive=True)

            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Input Text", 
                    lines=5,
                    value="The quick brown fox jumps over the lazy dog. It's a beautiful day in the neighborhood."
                )

        token_info = gr.Textbox(label="Token Information")
        highlighted_text = gr.HTML(label="Highlighted Text")
        prediction_plot = gr.Plot(label="Prediction Distributions")

        # Add a state to store the last clicked position
        last_click_position = gr.State(-1)

        # Handler for when text is clicked. NOTE: gradio passes in the evt based on type annotations
        def store_click_position(evt: gr.SelectData):
            return evt.index[0]

        # Handler for when sliders change
        def update_visualization(text: str, reference_steps: int, test_steps: int, char_index: int):
            if char_index == -1:  # No click has happened yet
                return highlighted_text.value, prediction_plot.value, token_info.value
            assert reference_steps >= test_steps
            visualizer.process_text(text, max_steps=reference_steps)
            return visualizer.visualize_predictions(char_index, reference_steps, test_steps)

        # Update test_steps maximum when reference_steps changes
        def update_test_steps_max(reference_value):
            ref_value = int(reference_value)
            return gr.update(maximum=ref_value, value=min(int(test_steps.value), ref_value))

        # Store the click position when text is clicked and then update visualization
        text_input.select(
            fn=store_click_position,
            outputs=[last_click_position],
        ).then(  # Chain the second callback to run after the first
            fn=update_visualization,
            inputs=[text_input, reference_steps, test_steps, last_click_position],
            outputs=[highlighted_text, prediction_plot, token_info],
        )
        
        # Update visualization when sliders change
        reference_steps.change(
            fn=update_test_steps_max,
            inputs=[reference_steps],
            outputs=[test_steps]
        ).then(
            fn=update_visualization,
            inputs=[text_input, reference_steps, test_steps, last_click_position],
            outputs=[highlighted_text, prediction_plot, token_info]
        )
        
        test_steps.change(
            fn=update_visualization,
            inputs=[text_input, reference_steps, test_steps, last_click_position],
            outputs=[highlighted_text, prediction_plot, token_info]
        )

    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()
