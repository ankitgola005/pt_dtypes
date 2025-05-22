import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

seed = 42
device = "hpu" if torch.hpu.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world_size", type=int, default=2, help="Total number of processes"
    )
    parser.add_argument(
        "--master_addr", type=str, default="localhost", help="IP address of master node"
    )
    parser.add_argument(
        "--master_port", type=str, default="12345", help="Port of master node"
    )
    parser.add_argument(
        "--num_epochs", "-n", type=int, default=10, help="Number of epochs to train."
    )
    parser.add_argument(
        "--amp_ratio",
        "-r",
        type=float,
        default=1.0,
        help="Ratio of bf16 vs fp32 autocast.",
    )
    parser.add_argument(
        "--amp_policy",
        "-p",
        default="bf16_first",
        help="How autocast dtype is spread during training.",
    )
    return parser.parse_args()


def setup_distributed(rank, world_size, master_addr="localhost", master_port=12345):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(
        "hccl" if device == "hpu" else "gloo", rank=rank, world_size=world_size
    )


def cleanup_distributed():
    dist.destroy_process_group()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 10))

    def forward(self, x):
        return self.net(x)


def get_log_dir_name(base_dir="runs"):
    os.makedirs(base_dir, exist_ok=True)
    existing = [
        d
        for d in os.listdir(base_dir)
        if d.startswith("run_") and d[len("run_") :].isdigit()
    ]
    run_ids = [int(d[len("run_") :]) for d in existing]
    next_id = max(run_ids, default=-2) + 1
    return os.path.join(base_dir, f"run_{next_id}", "tensorbaord"), os.path.join(
        base_dir, f"run_{next_id}", "profile"
    )


def train(rank, world_size, master_addr, master_port, num_epochs, amp_policy):
    setup_distributed(rank, world_size, master_addr, master_port)
    torch.manual_seed(seed)

    model = Net().to(device)
    ddp_model = DDP(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    tb_dir, profiler_dir = get_log_dir_name()
    if rank == 0:
        writer = SummaryWriter(tb_dir)

    with profile(
        schedule=schedule(wait=0, warmup=3, active=1, repeat=0),
        activities=[ProfilerActivity.CPU, ProfilerActivity.HPU],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir),
    ) as profiler:
        for epoch in range(num_epochs):
            use_amp = amp_policy[epoch]
            with torch.autocast(
                device_type=device, dtype=torch.bfloat16, enabled=use_amp
            ):
                inputs = torch.randn(64, 10, device=device)
                targets = torch.randint(
                    0, 10, size=(64,), dtype=torch.long, device=device
                )
                optimizer.zero_grad()
                logits = ddp_model(inputs)
                loss = loss_fn(logits, targets)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    preds = logits.argmax(dim=1)
                    accuracy = (preds == targets).float().mean()
                profiler.step()

            if rank == 0:
                writer.add_scalar("TrainLoss", loss.item(), epoch)
                writer.add_scalar("TrainAcc", accuracy, epoch)
    print(f"Finished Training. loss: {loss.item()}, acc: {accuracy}")
    cleanup_distributed()


def generate_amp_policy(
    num_epochs,
    amp_ratio,
    policy="shuffle",
    seed=seed,
):
    """
    Returns a list of booleans (True = use autocast, False = use FP32) for each epoch.
    This avoids graph breaks due to conditionals for selecting dtype.

    Args:
        num_epochs: Total number of epochs.
        amp_ratio: Fraction (0.0–1.0) of epochs to run in AMP. 1 use all bf16, 0 use all fp32.
        policy: One of 'shuffle', 'bf16_first', or 'fp32_first'.
        seed: seed.
    """
    # 1 = autocast bf16, 0 = fp32
    num_amp_epochs = int(num_epochs * amp_ratio)
    num_fp32_epochs = num_epochs - num_amp_epochs

    if policy == "bf16_first":
        return [True] * num_amp_epochs + [False] * num_fp32_epochs
    elif policy == "fp32_first":
        return [False] * num_fp32_epochs + [True] * num_amp_epochs
    elif policy == "shuffle":
        torch.manual_seed(seed)
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


if __name__ == "__main__":
    args = parse_args()
    print(globals())

    amp_policy = generate_amp_policy(
        args.num_epochs, args.amp_ratio, args.amp_policy, seed
    )
    mp.spawn(
        train,
        args=(
            args.world_size,
            args.master_addr,
            args.master_port,
            args.num_epochs,
            amp_policy,
        ),
        nprocs=args.world_size,
        join=True,
    )
