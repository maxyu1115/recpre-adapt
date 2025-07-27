import torch

def save_checkpoint(model, optimizer, lr_scheduler, epoch, model_name: str):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, f"{model_name}_{epoch}.pt2")

def load_checkpoint(model, optimizer, lr_scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    return checkpoint["epoch"]
