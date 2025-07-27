import sys
import os
import torch
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recpre.raven_modeling_minimal import RavenForCausalLM
from recpre_adapt.utils import update_huggingface_implementation
from recpre_adapt.data_loaders.red_pajama import RedPajamaPMD

import umap

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from recpre_adapt.data_loaders.red_pajama import RedPajamaPMD

    torch.manual_seed(42)

    # Set up logging
    log_file = "umap_latents.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    model: RavenForCausalLM = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")
    model.to("cuda", dtype=torch.bfloat16) # type: ignore
    model.save_latents = True

    update_huggingface_implementation(model)

    batch_size = 4
    seq_len = 256
    num_steps = 32
    pmd = RedPajamaPMD(model.device, tokenizer, batch_size, seq_len)

    latents = []
    x_val = []
    with torch.no_grad():
        for i in range(10):
            x, _ = pmd.get_batch("train")
            model.forward(x, attention_mask=None, num_steps=torch.tensor((num_steps,)))
            model_latents: torch.Tensor = torch.stack(model.latents, dim=2) # type: ignore
            latents.append(model_latents.detach())
            x_val.append(x.detach().cpu())

        del model
        torch.cuda.empty_cache()

        latents = torch.cat(latents, dim=0)
        x_val = torch.cat(x_val, dim=0)
        
        U, S, V = torch.pca_lowrank(latents.reshape(-1, latents.shape[-1]).float(), q=50, center=True)
        # Project
        latents_pca = latents @ V[:, :50]   # shape: (n_samples, 50)
        print(latents_pca.shape)

        # send data back to CPU numpy
        latents_pca_np = latents_pca.cpu().numpy()

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.1,
            metric='euclidean',
            random_state=42
        )
        embedding_2d_np = reducer.fit_transform(latents_pca_np)    # (n_samples, 2)

        import matplotlib.pyplot as plt

        pts = embedding_2d_np
        plt.scatter(pts[:,0], pts[:,1], s=5, alpha=0.6)
        plt.title("UMAP projection")
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.show()
