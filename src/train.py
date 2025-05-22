import argparse
import torch
import torch.multiprocessing as mp
from models.rand_model import RandModel, get_rand_dataloader
from utils.utils import get_amp_policy, SEED
from utils.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world_size", type=int, default=2, help="Total number of processes"
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


def main(rank, world_size, num_epochs, amp_policy):
    torch.manual_seed(SEED)
    model = RandModel()
    dataloader = get_rand_dataloader()

    trainer = Trainer(
        model, rank, world_size, amp_policy=amp_policy, base_logdir="runs"
    )
    trainer.init_profiler()
    trainer.fit(dataloader, num_epochs)


if __name__ == "__main__":
    args = parse_args()
    print(globals())

    torch.manual_seed(SEED)
    amp_policy = get_amp_policy(args.num_epochs, args.amp_ratio, args.amp_policy)
    mp.spawn(
        main,
        args=(
            args.world_size,
            args.num_epochs,
            amp_policy,
        ),
        nprocs=args.world_size,
        join=True,
    )
