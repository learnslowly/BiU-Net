"""After early-stopping training, find the 'best' checkpoint and copy it to a
numbered epoch filename so test.py can load it via config.epoch.
"""
import argparse
import os
import shutil
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True, help='run name (config.run)')
    p.add_argument('--model_dir', default='checkpoints')
    args = p.parse_args()

    best_path = os.path.join(args.model_dir, f"checkpoint_{args.run}_epoch_best.pth")
    if not os.path.exists(best_path):
        # fallback: highest-numbered checkpoint
        import glob
        candidates = glob.glob(os.path.join(args.model_dir, f"checkpoint_{args.run}_epoch_*.pth"))
        if not candidates:
            raise RuntimeError(f"no checkpoint found for run={args.run}")
        # extract epoch nums
        epochs = []
        for c in candidates:
            try:
                ep = int(c.rsplit('_epoch_', 1)[1].split('.pth')[0])
                epochs.append((ep, c))
            except ValueError:
                continue
        if not epochs:
            raise RuntimeError("no numbered checkpoints found")
        epochs.sort()
        best_ep, best_src = epochs[-1]
        print(f"NO best.pth; using latest numbered ckpt at epoch={best_ep}")
        # already at numbered filename; nothing to do
        print(f"BEST_EPOCH={best_ep}")
        return

    ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
    ep = int(ckpt.get('epoch', 0))
    print(f"best.pth epoch={ep}")

    dst = os.path.join(args.model_dir, f"checkpoint_{args.run}_epoch_{ep}.pth")
    if not os.path.exists(dst):
        shutil.copy(best_path, dst)
        print(f"copied best -> {dst}")
    else:
        print(f"{dst} already exists")
    print(f"BEST_EPOCH={ep}")


if __name__ == '__main__':
    main()
