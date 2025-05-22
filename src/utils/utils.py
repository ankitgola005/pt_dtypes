import os
import torch
import torch.distributed as dist

SEED = 42
DEVICE = "hpu" if torch.hpu.is_available() else "cpu"


def setup_distributed(rank, world_size, device=DEVICE):
    """Setup distributed pg."""
    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12345")
        dist.init_process_group(
            "hccl" if device == "hpu" else "gloo", rank=rank, world_size=world_size
        )
    else:
        raise RuntimeError(
            f"Trying to initialize the default process group twice. {dist.is_available()=}. Should be `True`. {dist.is_initialized()=}. Should be `False`."
        )


def destroy_distributed():
    """Destroy distributed pg."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_log_dir_name(base_dir="runs", log_dir_suffix=None):
    """Get a log dir name for current run. Avoid overwriting logs from previous runs."""
    os.makedirs(base_dir, exist_ok=True)
    existing = [
        d
        for d in os.listdir(base_dir)
        if d.startswith("run_") and d[len("run_") :].isdigit()
    ]
    run_ids = [int(d[len("run_") :]) for d in existing]
    next_id = max(run_ids, default=-1) + 1
    base_name = "run" if log_dir_suffix is None else os.path.join("run", log_dir_suffix)
    return os.path.join(
        base_dir, f"{base_name}_{next_id}", "tensorbaord"
    ), os.path.join(base_dir, f"{base_name}_{next_id}", "profile")


def get_amp_policy(
    num_epochs,
    amp_ratio,
    policy="shuffle",
):
    """
    Returns a list of booleans (True = use autocast, False = use FP32) for each epoch.
    This avoids graph breaks due to conditionals for selecting dtype.

    Args:
        num_epochs: Total number of epochs.
        amp_ratio: Fraction (0.0–1.0) of epochs to run in AMP. 1 use all bf16, 0 use all fp32.
        policy: One of 'shuffle', 'bf16_first', or 'fp32_first'.
    """
    # 1 = autocast bf16, 0 = fp32
    num_amp_epochs = int(num_epochs * amp_ratio)
    num_fp32_epochs = num_epochs - num_amp_epochs

    if policy == "bf16_first":
        return [True] * num_amp_epochs + [False] * num_fp32_epochs
    elif policy == "fp32_first":
        return [False] * num_fp32_epochs + [True] * num_amp_epochs
    elif policy == "shuffle":
        base = torch.cat(
            [
                torch.ones(num_amp_epochs, dtype=torch.bool),
                torch.zeros(num_fp32_epochs, dtype=torch.bool),
            ]
        )
        return base[torch.randperm(num_epochs)].tolist()
    else:
        raise ValueError(
            f"Unsupported policy: {policy}. Choose from 'bf16_first', 'fp32_first', or 'shuffle'."
        )
