#!/usr/bin/env python3
# prep_shard_amazon.py
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

RNG = np.random.default_rng(2025)

def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(p)
    # Amazon 2018 CSV uses commas and "vote" can have commas inside (e.g., "1,234")
    return pd.read_csv(p)

def normalize_amazon(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonicalize columns:
      user_id, review_text, rating, helpful_votes
    """
    need = ["reviewerID", "reviewText", "overall"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing required Amazon column: {c}")

    out = pd.DataFrame()
    out["user_id"]     = df["reviewerID"].astype(str)
    out["review_text"] = df["reviewText"].astype(str)
    out["rating"]      = pd.to_numeric(df["overall"], errors="coerce")

    # helpful votes → many files use 'vote' as string with commas; sometimes 'helpfulVotes'
    hv = None
    if "helpfulVotes" in df.columns:
        hv = pd.to_numeric(df["helpfulVotes"], errors="coerce")
    elif "vote" in df.columns:
        hv = pd.to_numeric(df["vote"].astype(str).str.replace(",", ""), errors="coerce")
    else:
        hv = pd.Series(0.0, index=df.index)

    out["helpful_votes"] = hv.fillna(0.0).clip(lower=0.0)
    out = out.dropna(subset=["user_id", "review_text", "rating"])
    out["review_text"] = out["review_text"].str.strip()
    out = out[out["review_text"].str.len() > 0]
    return out.reset_index(drop=True)

def split_indices(n, train=0.65, val=0.10, test=0.25):
    idx = np.arange(n)
    RNG.shuffle(idx)
    n_train = math.floor(train * n)
    n_val   = math.ceil(val * n)
    n_test  = n - n_train - n_val
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

def shard_per_user(df: pd.DataFrame, out_dir: Path,
                   label_mode: str = "sentiment-bin",
                   min_reviews: int = 2,
                   max_reviews: int = 1000,
                   cap_users: int | None = None,
                   drop_neutral: bool = True):
    """
    Writes per-user CSVs:
      train_<user>.csv, val_<user>.csv, test_<user>.csv
    Saves clients_meta.json with RF/RL split by 50% cumulative sample mass.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # create y_cls according to label_mode
    df = df.copy()
    if label_mode == "sentiment-bin":
        # 1 if rating >=4, 0 if rating <=2; drop 3 by default
        df["y_cls"] = df["rating"].apply(lambda r: 1 if r >= 4.0 else (0 if r <= 2.0 else None))
        if drop_neutral:
            df = df[df["y_cls"].notnull()].copy()
    elif label_mode == "sentiment-3":
        df["y_cls"] = df["rating"].round().astype(int).clip(1,5).map({1:0,2:0,3:1,4:2,5:2})
    elif label_mode == "rating-reg":
        df["y_cls"] = df["rating"].astype(float)
    else:
        raise ValueError("label_mode must be one of: sentiment-bin | sentiment-3 | rating-reg")

    # stabilize helpful: cap at p99 and log1p
    cap = np.nanpercentile(df["helpful_votes"].values, 99.0) if len(df) else 0.0
    df["helpful_votes"] = np.minimum(df["helpful_votes"].values, cap)
    df["helpful_log1p"] = np.log1p(df["helpful_votes"].values)

    # filter by user review counts
    sizes = df.groupby("user_id").size().rename("n").reset_index()
    sizes = sizes[(sizes["n"] >= min_reviews) & (sizes["n"] <= max_reviews)].copy()
    if cap_users is not None and len(sizes) > cap_users:
        sizes = sizes.sample(n=cap_users, random_state=2025)

    keep_users = set(sizes["user_id"].tolist())
    df = df[df["user_id"].isin(keep_users)].copy()

    # RF/RL split (50% cumulative)
    sizes = sizes.sort_values("n", ascending=False).reset_index(drop=True)
    cum = sizes["n"].cumsum()
    total = sizes["n"].sum()
    rf_users = sizes.loc[cum <= 0.5 * total, "user_id"].tolist()
    rl_users = sizes.loc[cum  > 0.5 * total, "user_id"].tolist()

    meta_users = {}
    cols = ["user_id", "review_text", "rating", "helpful_votes", "helpful_log1p", "y_cls"]
    for uid, udf in df.groupby("user_id"):
        udf = udf.reset_index(drop=True)
        tr, va, te = split_indices(len(udf), 0.65, 0.10, 0.25)

        def dump(split, idxs):
            part = udf.iloc[idxs][cols]
            (out_dir / f"{split}_{uid}.csv").write_text(part.to_csv(index=False))

        dump("train", tr)
        dump("val",   va)
        dump("test",  te)

        meta_users[uid] = {
            "n_total": int(len(udf)),
            "n_train": int(len(tr)),
            "n_val":   int(len(va)),
            "n_test":  int(len(te)),
            "is_rf": uid in rf_users
        }

    meta = {
        "dataset": "amazon",
        "label_mode": label_mode,
        "min_reviews": min_reviews,
        "max_reviews": max_reviews,
        "cap_users": cap_users,
        "n_users": len(meta_users),
        "n_total_samples": int(sizes["n"].sum()),
        "rf_users": rf_users,
        "rl_users": rl_users,
        "users": meta_users
    }
    (out_dir / "clients_meta.json").write_text(json.dumps(meta, indent=2))
    return meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="Amazon Reviews 2018 CSV (with reviewerID, reviewText, overall, vote/helpfulVotes)")
    ap.add_argument("--out_dir", default="data/amazon_clients", help="Output folder for per-client CSVs")
    ap.add_argument("--label_mode", choices=["sentiment-bin","sentiment-3","rating-reg"], default="sentiment-bin")
    ap.add_argument("--min_reviews", type=int, default=2)
    ap.add_argument("--max_reviews", type=int, default=1000)
    ap.add_argument("--cap_users", type=int, default=None, help="Optionally limit total users")
    ap.add_argument("--keep_neutral", action="store_true", help="Keep rating==3 in sentiment-bin (default: drop)")
    args = ap.parse_args()

    df = load_table(args.input_csv)
    df = normalize_amazon(df)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = shard_per_user(
        df,
        out_dir=out,
        label_mode=args.label_mode,
        min_reviews=args.min_reviews,
        max_reviews=args.max_reviews,
        cap_users=args.cap_users,
        drop_neutral=not args.keep_neutral
    )

    print(json.dumps({
        "out_dir": str(out),
        "n_users": meta["n_users"],
        "n_total_samples": meta["n_total_samples"],
        "rf_users": len(meta["rf_users"]),
        "rl_users": len(meta["rl_users"]),
    }, indent=2))

if __name__ == "__main__":
    main()


"""
python prep_shard_amazon.py \
  --input_csv /path/to/AMAZON_2018_reviews.csv \
  --out_dir data/amazon_clients \
  --label_mode sentiment-bin \
  --min_reviews 2 --max_reviews 300 \
  --cap_users 5000


this will create 

data/amazon_clients/
  train_<USERID>.csv
  val_<USERID>.csv
  test_<USERID>.csv
  clients_meta.json
"""