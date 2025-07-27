import os
import sys
import torch
import torch.nn as nn
from typing import Optional
import logging
from torch.optim.lr_scheduler import LambdaLR


# Add the parent directory to the Python path to make recpre importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recpre.raven_modeling_minimal import RavenConfig, RavenForCausalLM
from recpre_adapt.data_loaders import PoorMansDataLoaderBase
from recpre_adapt.utils import update_huggingface_implementation, get_lr_scheduler


class GatedMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None, output_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim * 2
        if output_dim is None:
            output_dim = input_dim
        self.fc = nn.Linear(input_dim, hidden_dim * 2)
        self.nonlin = nn.SiLU()
        self.proj = nn.Linear(hidden_dim, output_dim)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor):
        x_fc_1, x_fc_2 = self.fc(x).chunk(2, dim=-1)
        x = self.nonlin(x_fc_1) * x_fc_2
        return self.proj(x)


class LoopPredictor2(nn.Module):
    def __init__(self, config: RavenConfig, max_loops: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.mlp1 = GatedMLP(config.n_embd)
        self.norm2 = nn.LayerNorm(config.n_embd)
        self.mlp2 = GatedMLP(config.n_embd)
        self.out = nn.Linear(config.n_embd, max_loops)

        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor):
        x = self.norm1(x + self.mlp1(x))
        x = self.norm2(x + self.mlp2(x))
        return self.out(x).float()


class LoopPredictor(nn.Module):
    def __init__(self, config: RavenConfig, max_loops: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.mlp1 = GatedMLP(config.n_embd)
        # self.norm2 = nn.LayerNorm(config.n_embd)
        # self.mlp2 = GatedMLP(config.n_embd)
        self.out = nn.Linear(config.n_embd, max_loops)

        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor):
        x = self.norm1(x + self.mlp1(x))
        # x = self.norm2(x + self.mlp2(x))
        return self.out(x).float()


def compute_predict_loops_loss(
    model: RavenForCausalLM,
    loop_predictor: LoopPredictor,
    pmd: PoorMansDataLoaderBase,
    min_steps: int,
    max_steps: int,
    loops_per_eval: int = 1,
):
    with torch.no_grad():
        x, _ = pmd.get_batch("train")
        model.forward(x, attention_mask=None, num_steps=torch.tensor((max_steps,)))
        assert len(model.latents) == max_steps + 1
        latent_list: list[torch.Tensor] = model.latents
        # latents are of shape (num_steps, batch_size, seq_len, n_embd)
        latents = torch.stack(latent_list[min_steps::loops_per_eval], dim=0).to(dtype=loop_predictor.out.weight.dtype)
    # preds are of shape (num_steps, batch_size, seq_len, max_loops)
    preds = loop_predictor(latents)
    targets = torch.zeros_like(preds)
    for i in range(min_steps, max_steps + 1, loops_per_eval):
        offset_idx = i - min_steps
        targets[offset_idx // loops_per_eval, :, :, offset_idx // loops_per_eval] = 1.0
    loss = torch.nn.functional.cross_entropy(preds.view(-1, preds.shape[-1]), targets.view(-1, targets.shape[-1]))
    return loss


def train_loop_predictor(
    model: RavenForCausalLM,
    loop_predictor: LoopPredictor,
    pmd: PoorMansDataLoaderBase,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[LambdaLR] = None,
    min_steps: int = 0,
    max_steps: int = 32,
    loops_per_eval: int = 1,
    val_batchs: int = 10,
    save_prefix: str = "",
):
    x_val = []
    for i in range(val_batchs):
        x, _ = pmd.get_batch("train")
        x_val.append(x)

    loop_predictor.train()

    for epoch in range(num_epochs):
        x, _ = pmd.get_batch("train")
        optimizer.zero_grad()
        loss = compute_predict_loops_loss(model, loop_predictor, pmd, min_steps, max_steps, loops_per_eval=loops_per_eval)
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        if epoch % 10 == 9:
            loop_predictor.eval()
            with torch.no_grad():
                val_loss = 0
                for x in x_val:
                    loss = compute_predict_loops_loss(model, loop_predictor, pmd, min_steps, max_steps, loops_per_eval=loops_per_eval)
                    val_loss += loss.item()
                val_loss /= len(x_val)
                if lr_scheduler is not None:
                    logging.info(f"{save_prefix}: Epoch {epoch}, Val Loss {val_loss}, Current LR {lr_scheduler.get_last_lr()[0]}")
                else:
                    logging.info(f"{save_prefix}: Epoch {epoch}, Val Loss {val_loss}")
            loop_predictor.train()
        if epoch == 99 or epoch % 500 == 499:
            # Save the model
            torch.save(loop_predictor.state_dict(), f"{save_prefix}loop_predictor_{epoch}.pt")

    return loop_predictor


def train():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from recpre_adapt.data_loaders.red_pajama import RedPajamaPMD
    
    torch.manual_seed(42)

    # Set up logging
    log_file = "training_loop_predictor.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    model: RavenForCausalLM = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", trust_remote_code=True, code_revision="2a364bd96e3eaa831be324f7c1f9e74892e4e594")
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")
    model.to("cuda", dtype=torch.bfloat16) # type: ignore
    model.save_latents = True

    update_huggingface_implementation(model)

    batch_size = 4
    seq_len = 256
    pmd = RedPajamaPMD(model.device, tokenizer, batch_size, seq_len)

    min_steps = 0
    max_steps = 32
    loop_predictor = LoopPredictor(model.config, max_steps - min_steps + 1)
    loop_predictor.to(model.device, dtype=torch.bfloat16)
    
    max_epochs = 10000

    optimizer = torch.optim.AdamW(loop_predictor.parameters(), lr=0.001)
    lr_scheduler = get_lr_scheduler(optimizer, 100, max_epochs, "cosine")

    train_loop_predictor(model, loop_predictor, pmd, max_epochs, optimizer, lr_scheduler, min_steps=min_steps, max_steps=max_steps)


def train_sweep(loops_per_eval: int = 1):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from recpre_adapt.data_loaders.red_pajama import RedPajamaPMD

    torch.manual_seed(42)

    # Set up logging
    log_file = "training_loop_predictor.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    model: RavenForCausalLM = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", trust_remote_code=True, code_revision="2a364bd96e3eaa831be324f7c1f9e74892e4e594")
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")
    model.to("cuda", dtype=torch.bfloat16) # type: ignore
    model.save_latents = True

    update_huggingface_implementation(model)

    batch_size = 4
    seq_len = 256

    for step in range(0, 32, loops_per_eval):
        min_steps = step
        max_steps = step + loops_per_eval
        pmd = RedPajamaPMD(model.device, tokenizer, batch_size, seq_len)
        loop_predictor = LoopPredictor(model.config, 2)
        loop_predictor.to(model.device, dtype=torch.bfloat16)
        
        max_epochs = 500

        optimizer = torch.optim.AdamW(loop_predictor.parameters(), lr=0.001)

        train_loop_predictor(model, loop_predictor, pmd, max_epochs, optimizer, None, min_steps=min_steps, max_steps=max_steps, loops_per_eval=loops_per_eval, save_prefix=f"loops_{min_steps}vs{max_steps}")


if __name__ == "__main__":
    # train_sweep(loops_per_eval=1)
    train_sweep(loops_per_eval=4)
