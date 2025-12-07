# utils_noniid.py
import numpy as np
import torch
from torch.utils.data import Subset

# ---------- existing helpers ----------
def _to_numpy_1d(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    x = np.asarray(x).reshape(-1)
    if x.dtype.kind not in ("i", "u"):
        x = x.astype(int)
    return x

def set_seed(seed: int = 0):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def build_subsets(dataset, client_indices):
    return [Subset(dataset, idx.tolist()) for idx in client_indices]

# ---------- new: Non-IID + size-imbalanced partitioner ----------
def imbalanced_dirichlet_partition(
    targets,
    n_clients: int,
    alpha: float = 0.3,
    min_size: int = 20,
    seed: int = 0,
    # choose one of the following to control client sizes:
    target_sizes: np.ndarray | None = None,       # exact per-client counts (length n_clients)
    client_size_weights: np.ndarray | None = None,# will be normalized to total N
    lognormal_params: tuple[float, float] = (3.0, 1.0),  # (mean, sigma) for log-normal if neither of above provided
):
    """
    Create a label–non-IID and client–size-imbalanced split.

    Args
    ----
    targets : array-like of int labels (len = N)
    n_clients : number of clients
    alpha : Dirichlet concentration for label skew (smaller => more non-IID)
    min_size : minimum samples per client after partition
    seed : RNG seed
    target_sizes : optional exact per-client sample counts (sum must equal N)
    client_size_weights : optional positive weights; we scale to sum N
    lognormal_params : (mean, sigma) used to sample sizes if neither target_sizes nor weights are given

    Returns
    -------
    list[np.ndarray] : indices per client (length n_clients)
    """
    rng = np.random.RandomState(seed)
    y = _to_numpy_1d(targets)
    N = y.size
    n_classes = int(y.max()) + 1

    # ---- 1) Decide how many samples each client should have (size imbalance) ----
    if target_sizes is not None:
        sizes = np.asarray(target_sizes, dtype=int).copy()
        assert sizes.shape == (n_clients,), "target_sizes must have shape (n_clients,)"
        assert sizes.sum() == N, "sum(target_sizes) must equal number of samples"
        assert (sizes >= 0).all()
    elif client_size_weights is not None:
        w = np.asarray(client_size_weights, dtype=float).clip(min=1e-12)
        assert w.shape == (n_clients,), "client_size_weights must have shape (n_clients,)"
        sizes = np.floor(w / w.sum() * N).astype(int)
        # fix rounding to match N exactly
        while sizes.sum() < N:
            sizes[np.argmax(w / w.sum() * N - sizes)] += 1
        while sizes.sum() > N:
            sizes[np.argmax(sizes)] -= 1
    else:
        mu, sigma = lognormal_params
        draw = rng.lognormal(mean=mu, sigma=sigma, size=n_clients)
        sizes = np.floor(draw / draw.sum() * N).astype(int)
        # fix rounding
        while sizes.sum() < N:
            sizes[rng.randint(0, n_clients)] += 1
        while sizes.sum() > N:
            i = np.argmax(sizes)
            if sizes[i] > 0: sizes[i] -= 1

    # Enforce min_size by borrowing from the largest clients
    if min_size > 0:
        need = np.where(sizes < min_size)[0]
        give = np.where(sizes > min_size)[0]
        for i in need:
            deficit = min_size - sizes[i]
            while deficit > 0 and give.size > 0:
                j = give[np.argmax(sizes[give])]
                take = min(deficit, sizes[j] - min_size)
                if take <= 0:
                    give = give[sizes[give] - min_size > 0]
                    if give.size == 0: break
                    continue
                sizes[i] += take
                sizes[j] -= take
                deficit -= take

    # Final sanity
    assert sizes.sum() == N, "internal size bookkeeping error"
    assert (sizes >= 0).all()

    # ---- 2) Per-class Dirichlet to get non-IID label preference ----
    # We'll assign whole indices while respecting each client's remaining quota.
    rem = sizes.copy()
    client_bins = [[] for _ in range(n_clients)]

    for k in range(n_classes):
        idx_k = np.where(y == k)[0]
        rng.shuffle(idx_k)
        remaining_k = len(idx_k)
        start = 0

        if remaining_k == 0:
            continue

        # base Dirichlet over clients
        p = rng.dirichlet(alpha * np.ones(n_clients))
        # bias by remaining capacity so we don't overflow quotas
        score = p * (rem + 1e-12)

        # Greedy fill: highest score gets filled first, capped by its remaining quota
        order = np.argsort(-score)
        for i in order:
            if remaining_k <= 0:
                break
            if rem[i] <= 0:
                continue
            take = min(rem[i], remaining_k)
            if take > 0:
                client_bins[i].extend(idx_k[start:start + take].tolist())
                start += take
                rem[i] -= take
                remaining_k -= take

        # If some class samples remain (e.g., all rem reached 0 due to min_size constraints),
        # spill them into clients with any leftover capacity.
        if remaining_k > 0 and rem.sum() > 0:
            spill_order = np.argsort(-rem)
            for i in spill_order:
                if remaining_k <= 0:
                    break
                if rem[i] <= 0:
                    continue
                take = min(rem[i], remaining_k)
                client_bins[i].extend(idx_k[start:start + take].tolist())
                start += take
                rem[i] -= take
                remaining_k -= take

        # Final safeguard: if still remaining (shouldn't happen), dump to the largest client
        if remaining_k > 0:
            i = np.argmax(sizes)
            client_bins[i].extend(idx_k[start:].tolist())
            remaining_k = 0

    # Convert to arrays and sort indices per client (optional)
    client_indices = [np.array(sorted(b)) for b in client_bins]
    # Asserts (can be commented out for speed)
    assert sum(len(b) for b in client_indices) == N
    for i, cnt in enumerate([len(b) for b in client_indices]):
        assert cnt == sizes[i], f"client {i} got {cnt}, expected {sizes[i]}"
    return client_indices
