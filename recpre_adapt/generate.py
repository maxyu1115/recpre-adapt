import torch
from transformers import AutoModelForCausalLM

from recpre.raven_modeling_minimal import RavenForCausalLM
from recpre_adapt.raven_exit_model import RavenAdaptiveModel, RavenExitModel

def load_model(model_path: str):
    model: RavenForCausalLM = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", trust_remote_code=True)
    exit_model = RavenExitModel(model.config)
    exit_model.load_state_dict(torch.load(model_path + "/exit_model.pt"))
    return RavenAdaptiveModel.from_models(model, exit_model)
