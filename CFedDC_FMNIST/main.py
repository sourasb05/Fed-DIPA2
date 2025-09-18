# main_cfeddc.py
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split

from server_cfeddc import CFedDCServer
from utils_noniid import set_seed

def main():
    p = argparse.ArgumentParser()
    # data / partition
    p.add_argument("--num_clients", type=int, default=50)
    p.add_argument("--alpha", type=float, default=0.3, help="Dirichlet concentration")
    p.add_argument("--seed", type=int, default=0)
    # training
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--frac", type=float, default=1.0, help="fraction of clients per round")
    p.add_argument("--local_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    # CFedDC
    p.add_argument("--n_clusters", type=int, default=3)
    p.add_argument("--kappa", type=float, default=1.0, help="fraction of RF per round")
    p.add_argument("--delta", type=float, default=1.0, help="fraction of RL per round")
    p.add_argument("--lam_min", type=float, default=0.4)
    p.add_argument("--lam_max", type=float, default=0.6)
    p.add_argument("--tau", type=float, default=0.5)
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- DATA: Fashion-MNIST ---
    tfm = transforms.Compose([transforms.ToTensor()])
    root = "./data"
    train_full = datasets.FashionMNIST(root, train=True,  download=True, transform=tfm)
    test_set  = datasets.FashionMNIST(root, train=False, download=True, transform=tfm)

    # global validation split from full train
    n_val = 5000
    n_train = len(train_full) - n_val
    train_set, val_set = random_split(
        train_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    server = CFedDCServer(
        dataset_train=train_full,   # pass full to let partitioner index into original targets
        dataset_val=val_set,
        dataset_test=test_set,
        num_clients=args.num_clients,
        alpha=args.alpha,
        seed=args.seed,
        batch_size=args.batch_size,
        lr=args.lr,
        local_epochs=args.local_epochs,
        rounds=args.rounds,
        frac=args.frac,
        n_clusters=args.n_clusters,
        kappa=args.kappa,
        delta=args.delta,
        lam_min=args.lam_min,
        lam_max=args.lam_max,
        tau=args.tau,
        device=device,
    )

    server.run()

if __name__ == "__main__":
    main()
