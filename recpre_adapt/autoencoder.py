import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch.optim.lr_scheduler import LambdaLR

# Add the parent directory to the Python path to make recpre importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recpre.raven_modeling_minimal import RavenForCausalLM
from recpre_adapt.data_loaders import PoorMansDataLoaderBase
from recpre_adapt.utils import update_huggingface_implementation, get_lr_scheduler


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        A simple autoencoder.

        Args:
            input_dim: Dimensionality of the input and output. This would typically
                       be the dimension of the latent states from RavenForCausalLM
                       (e.g., config.n_embd).
            hidden_dim: Dimensionality of the hidden representation (bottleneck).
                        For dimensionality reduction, this is typically smaller
                        than input_dim.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.encoder_fc = nn.Linear(input_dim, hidden_dim)
        self.decoder_fc = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.ReLU() # Or nn.Sigmoid(), nn.Tanh(), etc.
        # self.activation = nn.Tanh()

    @staticmethod
    def load_from_checkpoint(checkpoint_path: str):
        state_dict = torch.load(checkpoint_path)
        input_dim = state_dict["encoder_fc.weight"].shape[1]
        hidden_dim = state_dict["encoder_fc.weight"].shape[0]
        autoencoder = Autoencoder(input_dim, hidden_dim)
        autoencoder.load_state_dict(state_dict)
        return autoencoder

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the autoencoder.

        Args:
            x: Input tensor. Expected shape: (..., input_dim), where ...
               can be batch_size, sequence_length, etc.

        Returns:
            A tuple containing:
            - reconstructed_x (torch.Tensor): The autoencoder's reconstruction of x.
                                              Shape: (..., input_dim)
            - features (torch.Tensor): The hidden features after activation.
                                       Shape: (..., hidden_dim)
            - reconstruction_loss (torch.Tensor): MSE reconstruction loss.
        """
        # Encode
        features = self.encode(x)

        # Decode
        reconstructed_x = self.decode(features)

        # Reconstruction loss (Mean Squared Error)
        reconstruction_loss = F.mse_loss(reconstructed_x, x)

        return reconstructed_x, features, reconstruction_loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input tensor into hidden features.

        Args:
            x: Input tensor. Shape: (..., input_dim)

        Returns:
            torch.Tensor: Hidden features after activation. Shape: (..., hidden_dim)
        """
        current_input_dim = x.shape[-1]

        if current_input_dim != self.input_dim:
            raise ValueError(
                f"Input tensor's last dimension ({current_input_dim}) "
                f"does not match model's input_dim ({self.input_dim})"
            )

        encoded_activations = self.encoder_fc(x)
        features = self.activation(encoded_activations)

        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decodes hidden features back into the original input space.

        Args:
            features: Hidden features. Shape: (..., hidden_dim)

        Returns:
            torch.Tensor: Reconstructed tensor. Shape: (..., input_dim)
        """
        current_hidden_dim = features.shape[-1]

        if current_hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Feature tensor's last dimension ({current_hidden_dim}) "
                f"does not match model's hidden_dim ({self.hidden_dim})"
            )

        reconstructed_x = self.decoder_fc(features)

        return reconstructed_x


class SparseAutoencoder(Autoencoder):
    def __init__(self, input_dim: int, hidden_dim: int, l1_coeff: float = 0.001):
        """
        A simple sparse autoencoder.

        Args:
            input_dim: Dimensionality of the input and output. This would typically
                       be the dimension of the latent states from RavenForCausalLM
                       (e.g., config.n_embd).
            hidden_dim: Dimensionality of the sparse hidden representation.
                        For sparse autoencoders, this is often larger than input_dim
                        (e.g., 4 * input_dim).
            l1_coeff: Coefficient for the L1 sparsity penalty on the hidden layer
                      activations.
        """
        super().__init__(input_dim, hidden_dim)
        self.l1_coeff = l1_coeff
        self.activation = nn.Tanh()

        # Optional: Initialize biases for the encoder to negative values
        # to encourage sparsity from the beginning, though good initialization
        # of weights (like Kaiming for ReLU) is often sufficient.
        # with torch.no_grad():
        #     self.encoder_fc.bias.uniform_(-0.1, -0.01)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the sparse autoencoder.

        Args:
            x: Input tensor. Expected shape: (..., input_dim), where ...
               can be batch_size, sequence_length, etc.

        Returns:
            A tuple containing:
            - reconstructed_x (torch.Tensor): The autoencoder's reconstruction of x.
                                              Shape: (..., input_dim)
            - features (torch.Tensor): The sparse hidden features after ReLU.
                                       Shape: (..., hidden_dim)
            - total_loss (torch.Tensor): The total loss (reconstruction + sparsity).
            - reconstruction_loss (torch.Tensor): MSE reconstruction loss.
            - sparsity_loss (torch.Tensor): L1 sparsity loss on features.
        """
        reconstructed_x, features, reconstruction_loss = super().forward(x)

        # Sparsity loss (L1 penalty on feature activations)
        # We calculate the L1 norm of each feature vector in the batch/sequence
        # and then average these norms.
        sparsity_loss = self.l1_coeff * torch.linalg.norm(features, ord=1, dim=-1).mean()
        # An alternative, often used, is the mean absolute value of all feature activations:
        # sparsity_loss = self.l1_coeff * features.abs().mean()

        total_loss = reconstruction_loss + sparsity_loss

        return reconstructed_x, features, total_loss


def compute_autoencoder_loss(
    model: RavenForCausalLM,
    autoencoder: Autoencoder,
    pmd: PoorMansDataLoaderBase,
    num_steps: int,
):
    with torch.no_grad():
        x, _ = pmd.get_batch("train")
        model.forward(x, attention_mask=None, num_steps=torch.tensor((num_steps,)))
        assert len(model.latents) == num_steps + 1
        latent_list: list[torch.Tensor] = model.latents
        latents = torch.stack(latent_list, dim=2).to(dtype=autoencoder.decoder_fc.weight.dtype)
    reconstructed_x, features, loss = autoencoder(latents)
    return loss


def train_autoencoder(
    model: RavenForCausalLM,
    autoencoder: Autoencoder,
    pmd: PoorMansDataLoaderBase,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LambdaLR,
    num_steps: int = 32,
    val_batchs: int = 10,
):
    x_val = []
    for i in range(val_batchs):
        x, _ = pmd.get_batch("train")
        x_val.append(x)

    autoencoder.train()

    for epoch in range(num_epochs):
        x, _ = pmd.get_batch("train")
        optimizer.zero_grad()
        loss = compute_autoencoder_loss(model, autoencoder, pmd, num_steps)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if epoch % 10 == 9:
            autoencoder.eval()
            with torch.no_grad():
                val_loss = 0
                for x in x_val:
                    loss = compute_autoencoder_loss(model, autoencoder, pmd, num_steps)
                    val_loss += loss.item()
                val_loss /= len(x_val)
                logging.info(f"Epoch {epoch}, Val Loss: {val_loss}, Current LR: {lr_scheduler.get_last_lr()[0]}")
            autoencoder.train()
        if epoch == 99 or epoch % 500 == 499:
            # Save the model
            torch.save(autoencoder.state_dict(), f"autoencoder_{epoch}.pt")


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from recpre_adapt.data_loaders.red_pajama import RedPajamaPMD
    
    torch.manual_seed(42)

    # Set up logging
    log_file = "training_autoencoder.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    model: RavenForCausalLM = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")
    model.to("cuda", dtype=torch.bfloat16) # type: ignore
    model.save_latents = True
    
    update_huggingface_implementation(model)

    batch_size = 8
    seq_len = 256
    pmd = RedPajamaPMD(model.device, tokenizer, batch_size, seq_len)
    autoencoder = Autoencoder(model.config.n_embd, 1024)
    # autoencoder = SparseAutoencoder(model.config.n_embd, model.config.n_embd * 4, l1_coeff=0.001)
    # autoencoder = SparseAutoencoder(model.config.n_embd, model.config.n_embd * 2, l1_coeff=0.001)
    autoencoder.to(model.device, dtype=torch.bfloat16)
    
    max_epochs = 20000

    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=0.001)
    lr_scheduler = get_lr_scheduler(optimizer, 1000, max_epochs, "cosine")

    train_autoencoder(model, autoencoder, pmd, max_epochs, optimizer, lr_scheduler)
