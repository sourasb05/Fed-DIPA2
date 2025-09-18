
# main_movielens.py
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from server_cfeddc import CFedDCServer
from utils_nonIID import partition_users_imbalanced, build_client_interactions
import os
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_clients", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=500)
    ap.add_argument("--local_epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--n_clusters", type=int, default=2)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--lam_min", type=float, default=0.3)
    ap.add_argument("--lam_max", type=float, default=0.7)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_json", type=str, default="results_movielens_cfeddc.json",
                    help="Path to save JSON results")

    args = ap.parse_args()
    cwd = os.getcwd()

    output_json = cwd + "/results/" + args.output_json

    ratings_df = pd.read_csv(cwd + "/data/ratings.dat", sep="::", engine="python",
                 names=["user_id", "item_id", "rating", "timestamp"])  # expect columns: user_id,item_id,rating[,timestamp]
    print(f"Ratings data: {ratings_df.shape[0]} ratings from {ratings_df['user_id'].nunique()} users on {ratings_df['item_id'].nunique()} items")
    
    # Encode users/items
    user_enc, item_enc = LabelEncoder(), LabelEncoder()
    ratings_df["user_id"] = user_enc.fit_transform(ratings_df["user_id"])
    ratings_df["item_id"] = item_enc.fit_transform(ratings_df["item_id"])
    n_users, n_items = len(user_enc.classes_), len(item_enc.classes_)

    # 3) Global train/val/test split
    ratings_df = ratings_df.sample(frac=1.0, random_state=args.seed)
    n = len(ratings_df); n_test = int(0.1*n); n_val = int(0.1*n)
    df_test = ratings_df.iloc[:n_test]
    df_val  = ratings_df.iloc[n_test:n_test+n_val]
    df_train= ratings_df.iloc[n_test+n_val:]

    # 4) Client partitions from TRAIN
    client_users = partition_users_imbalanced(df_train, n_clients=args.num_clients, seed=args.seed)
    client_dfs = build_client_interactions(df_train, client_users)

    client_splits = []
    for cid, cdf in enumerate(client_dfs):
        if len(cdf) < 50: 
            continue
        m = len(cdf); m_val = max(10, int(0.1*m))
        cdf = cdf.sample(frac=1.0, random_state=args.seed+cid)
        cdf_train = cdf.iloc[m_val:]
        cdf_val   = cdf.iloc[:m_val]
        client_splits.append((cdf_train, cdf_val))

    # 5) Instantiate server
    server = CFedDCServer(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        n_users=n_users,
        n_items=n_items,
        num_clients=args.num_clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_clusters=args.n_clusters,
        kappa=args.kappa,
        delta=args.delta,
        lam_min=args.lam_min,
        lam_max=args.lam_max,
        tau=args.tau,
        seed=args.seed,
        output_json=output_json,
        client_splits=client_splits
    )

    server.run()

if __name__ == "__main__":
    main()
