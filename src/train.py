import argparse
import torch
import torch.multiprocessing as mp
from models.rand_model import RandModel, get_rand_dataloader
from utils.utils import get_amp_policy, get_log_dir_name, SEED
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


def main(rank, world_size, num_epochs, amp_policy, amp_ratio, amp_policy_list):
    torch.manual_seed(SEED)
    model = RandModel()
    dataloader = get_rand_dataloader()
    base_log_dir = get_log_dir_name(
        base_dir="runs",
        log_dir_suffix=f"epochs_{num_epochs}_policy_{amp_policy}_ratio_{amp_ratio}",
    )

    trainer = Trainer(
        model,
        rank,
        world_size,
        amp_policy_list=amp_policy_list,
        base_log_dir=base_log_dir,
        num_epochs=num_epochs,
        print_freq=1000,
    )
    trainer.init_profiler()
    trainer.fit(dataloader, accuracy=True)


if __name__ == "__main__":
    args = parse_args()
    print(globals())

    torch.manual_seed(SEED)
    amp_policy_list = get_amp_policy(args.num_epochs, args.amp_ratio, args.amp_policy)
    mp.spawn(
        main,
        args=(
            args.world_size,
            args.num_epochs,
            args.amp_policy,
            args.amp_ratio,
            amp_policy_list,
        ),
        nprocs=args.world_size,
        join=True,
    )
