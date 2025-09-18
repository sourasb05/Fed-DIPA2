# --- MovieLens partitioning utilities ---

import numpy as np
import pandas as pd

def partition_users_imbalanced(
    ratings_df: pd.DataFrame,
    n_clients: int,
    seed: int = 0,
    # control imbalance of client sizes (number of users per client):
    lognormal_mean: float = 2.0,
    lognormal_sigma: float = 1.0,
):
    """
    Group users into `n_clients` imbalanced buckets.
    Returns: dict client_id -> set(user_ids)
    """
    rng = np.random.RandomState(seed)
    users = ratings_df["user_id"].unique()
    n_users = len(users)

    # draw client "capacities" (how many users per client) from log-normal
    raw = rng.lognormal(mean=lognormal_mean, sigma=lognormal_sigma, size=n_clients)
    caps = np.floor(raw / raw.sum() * n_users).astype(int)
    # fix rounding
    while caps.sum() < n_users: caps[rng.randint(0, n_clients)] += 1
    while caps.sum() > n_users:
        i = caps.argmax()
        if caps[i] > 0: caps[i] -= 1

    rng.shuffle(users)
    client_users = {}
    start = 0
    for cid in range(n_clients):
        end = start + caps[cid]
        client_users[cid] = set(users[start:end])
        start = end
    return client_users

def build_client_interactions(ratings_df: pd.DataFrame, client_users: dict):
    """
    Slice the ratings by the user groups. Returns list of dataframes per client.
    """
    client_dfs = []
    for cid in range(len(client_users)):
        df = ratings_df[ratings_df["user_id"].isin(client_users[cid])].copy()
        client_dfs.append(df.reset_index(drop=True))
    return client_dfs
