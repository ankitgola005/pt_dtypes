from contextlib import nullcontext
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.utils import (
    DEVICE,
    get_log_dir_name,
    setup_distributed,
    destroy_distributed,
    rank_zero_print,
)


class Trainer:
    def __init__(
        self,
        model,
        rank,
        world_size,
        num_epochs=1,
        lr=1e-3,
        device=DEVICE,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.SGD,
        amp_policy_list=None,
        base_logdir=".",
        log_dir_suffix=None,
        print_freq=100,
    ):
        self.device = device
        self.rank = rank
        self.model = model.to(self.device)
        self.criterion = criterion(reduction="mean")
        self.optimizer = optimizer
        self.lr = lr
        self.num_epochs = num_epochs
        self.amp_policy_list = amp_policy_list
        self.profiler = None
        self.profiler_enabled = False
        self.profiler_context = nullcontext()
        self.logdir, self.profiler_dir = get_log_dir_name(
            base_dir=base_logdir, log_dir_suffix=log_dir_suffix
        )
        self.writer = SummaryWriter(log_dir=self.logdir) if self.rank == 0 else None
        self.world_size = world_size
        self.print_freq = print_freq

    def init_profiler(self, schedule=None, profiler_dir=None):
        self.profiler_enabled = True
        self.profiler_dir = (
            profiler_dir if profiler_dir is not None else self.profiler_dir
        )
        schedule = (
            schedule
            if schedule is not None
            else torch.profiler.schedule(wait=3, warmup=3, active=3, repeat=0)
        )
        self.profiler_context = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.HPU,
            ],
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.profiler_dir),
        )

    def get_accuracy(self, logits, targets):
        preds = torch.argmax(logits, dim=1)
        correct = (preds == targets).sum().item()
        return correct

    def train_step(self, batch, amp_enabled=False):
        self.model.train()
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        self.optimizer.zero_grad()
        with torch.autocast(
            device_type=self.device, dtype=torch.bfloat16, enabled=amp_enabled
        ):
            logits = self.model(x)
            loss = self.criterion(logits, y)
        loss.backward()
        self.optimizer.step()
        return loss, logits, y

    def train_epoch(self, dataloader, epoch_idx, accuracy=False, amp_enabled=False):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with self.profiler_context as prof:
            for batch in dataloader:
                loss, logits, targets = self.train_step(batch, amp_enabled)

                with torch.no_grad():
                    total_loss += loss * targets.size(0)
                    total_samples += targets.size(0)
                    if accuracy:
                        total_correct += self.get_accuracy(logits, targets)
                if self.profiler_enabled:
                    prof.step()

        avg_loss = total_loss / total_samples
        if self.rank == 0:
            self.writer.add_scalar("TrainLoss", loss.item(), epoch_idx)
        if accuracy:
            accuracy = total_correct / total_samples * 100
            if self.rank == 0:
                self.writer.add_scalar("TrainAcc", accuracy, epoch_idx)
            return avg_loss, accuracy

        return avg_loss, None

    def fit(self, dataloader, accuracy=False):
        if self.world_size > 1:
            setup_distributed(self.rank, self.world_size)
            self.model = DDP(self.model)
        self.optimizer = self.optimizer(self.model.parameters(), self.lr)

        rank_zero_print(self.rank, "Starting fit")
        for epoch in range(self.num_epochs):
            amp_enabled = self.amp_policy_list[epoch]
            loss, acc = self.train_epoch(dataloader, epoch, accuracy, amp_enabled)
            log = f"Epoch {epoch}: Loss = {loss:.6f}"
            if accuracy:
                log += f", Accuracy = {acc:.6f}%"
            if epoch % self.print_freq == 0:
                rank_zero_print(self.rank, log)

        if self.world_size > 1:
            destroy_distributed()
