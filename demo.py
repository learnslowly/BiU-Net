import os
import torch
import torch.distributed as dist
import sys

def main():
    # Setup distributed environment from SLURM
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))

    # Use cuda:0 because SLURM with --gpus-per-task=1 ensures only 1 visible GPU per task
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize process group
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size
    )

    # Create a tensor for signaling stop
    should_stop = torch.tensor([0], dtype=torch.int, device=device)

    # Rank 0 sets the stop signal
    if rank == 0:
        print(f"[Rank {rank}] Setting should_stop = 1 and broadcasting.")
        should_stop.fill_(1)

    # Sync all ranks and broadcast the stop signal
    dist.barrier()
    dist.broadcast(should_stop, src=0)

    print(f"[Rank {rank}] Received should_stop = {should_stop.item()}")

    # Exit all ranks cleanly if stop signal was received
    if should_stop.item() == 1:
        print(f"[Rank {rank}] Exiting after receiving stop signal.")
        dist.destroy_process_group()
        sys.exit(0)

    # Optional: Add more logic here if needed before exiting
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
