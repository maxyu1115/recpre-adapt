import torch


def _compute_vector_norm(latents: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(latents[:, :, 1:, :], ord=2, dim=-1)

def _compute_incremental_latent_distances(latents: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(latents[:, :, 1:, :] - latents[:, :, :-1, :], ord=2, dim=-1)

def _compute_latent_distances(latents: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(latents[:, :, 1:, :] - latents[:, :, 0, :].unsqueeze(2), ord=2, dim=-1)

def _compute_incremental_cosine_latent_distances(latents: torch.Tensor) -> torch.Tensor:
    return torch.cosine_similarity(latents[:, :, 1:, :], latents[:, :, :-1, :], dim=-1)

def _compute_cosine_latent_distances(latents: torch.Tensor) -> torch.Tensor:
    return torch.cosine_similarity(latents[:, :, 1:, :], latents[:, :, 0, :].unsqueeze(2), dim=-1)


def compute_feature_vectors(latents: torch.Tensor) -> torch.Tensor:
    vector_norm = _compute_vector_norm(latents)
    incremental_latent_distances = _compute_incremental_latent_distances(latents)
    latent_distances = _compute_latent_distances(latents)
    incremental_cosine_latent_distances = _compute_incremental_cosine_latent_distances(latents)
    cosine_latent_distances = _compute_cosine_latent_distances(latents)
    return torch.stack([
        vector_norm,
        incremental_latent_distances,
        latent_distances,
        incremental_cosine_latent_distances,
        cosine_latent_distances,
    ], dim=-1)
