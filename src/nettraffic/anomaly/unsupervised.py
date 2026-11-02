"""Model-free detectors on profile-normalised deviations.

Both detectors look at ``dev = log1p(y) - profile[hour_of_week]`` per metric,
where the profile is learned on the pre-detection period. They are independent
of the forecaster, so they still work when the forecaster drifts, and they see
cross-metric structure (e.g. internet up while calls are flat).

* Isolation Forest on short feature windows (current + 3h mean deviation).
* LSTM autoencoder (PyTorch) over 24h multivariate windows; the reconstruction
  error of the latest step is the score (online-compatible: only past data).

Scores are turned into tail probabilities against their own distribution on
held-out *training* data, so they are comparable with the residual detector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest

from nettraffic.anomaly.tail import TailModel
from nettraffic.features import FeatureContext, hour_of_week_profile
from nettraffic.panel import Panel

log = logging.getLogger(__name__)
CLIP = 4.0


def deviations(panel: Panel, ctx: FeatureContext, train_stop: int) -> np.ndarray:
    """(C, T, M) log deviation from the hour-of-week profile learned before train_stop."""
    how = ctx.cal.hour_of_week[: panel.n_times]
    out = np.empty(panel.values.shape, dtype=np.float32)
    for k, m in enumerate(panel.metrics):
        logy = np.log1p(panel.metric(m).astype(np.float64))
        prof = hour_of_week_profile(logy, how, train_stop)
        out[:, :, k] = np.clip(logy - prof[:, how], -CLIP, CLIP)
    return out


# --------------------------------------------------------------------------- Isolation Forest
def _if_features(dev: np.ndarray, ctx: FeatureContext, times: np.ndarray) -> np.ndarray:
    d = np.nan_to_num(dev, nan=0.0)
    cs = np.cumsum(np.concatenate([np.zeros_like(d[:, :1]), d], axis=1), axis=1)
    s = np.clip(times - 2, 0, None)
    roll3 = (cs[:, times + 1] - cs[:, s]) / (times + 1 - s)[None, :, None]
    cur = d[:, times]
    hour = ctx.cal.hour[times]
    hs = np.broadcast_to(np.sin(2 * np.pi * hour / 24)[None, :, None], (*cur.shape[:2], 1))
    hc = np.broadcast_to(np.cos(2 * np.pi * hour / 24)[None, :, None], (*cur.shape[:2], 1))
    off = np.broadcast_to(ctx.cal.is_offday[times][None, :, None].astype(float), (*cur.shape[:2], 1))
    return np.concatenate([cur, roll3, hs, hc, off], axis=2)  # (C, W, 2M+3)


@dataclass
class IForestDetector:
    seed: int = 42
    n_train: int = 120_000

    def run(
        self, dev: np.ndarray, ctx: FeatureContext, train_times: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        Xtr = _if_features(dev, ctx, train_times).reshape(-1, dev.shape[2] * 2 + 3)
        idx = rng.permutation(len(Xtr))
        n_null = min(60_000, len(idx) // 3)  # held-out rows -> null score distribution
        null_idx, fit_idx = idx[:n_null], idx[n_null : n_null + self.n_train]
        model = IsolationForest(n_estimators=200, max_samples=4096, random_state=self.seed, n_jobs=-1)
        model.fit(Xtr[fit_idx])
        tail = TailModel.fit(-model.score_samples(Xtr[null_idx]))
        Xd = _if_features(dev, ctx, times)
        c, w, f = Xd.shape
        score = -model.score_samples(Xd.reshape(-1, f)).reshape(c, w)
        surprisal = tail.surprisal(score)
        # attribute the cell-level score to the metrics that actually deviate
        d = np.abs(np.nan_to_num(dev[:, times], nan=0.0))
        share = d / np.maximum(d.max(axis=2, keepdims=True), 1e-6)
        return surprisal[:, :, None] * share


# --------------------------------------------------------------------------- LSTM autoencoder
@dataclass
class LSTMAutoencoderDetector:
    window: int = 24
    hidden: int = 32
    epochs: int = 4
    batch_size: int = 256
    n_train: int = 40_000
    seed: int = 42

    def run(self, dev: np.ndarray, train_times: np.ndarray, times: np.ndarray) -> np.ndarray | None:
        try:
            import torch
            from torch import nn
        except ImportError:  # optional dependency
            log.warning("torch not installed: LSTM autoencoder detector skipped")
            return None

        torch.manual_seed(self.seed)
        torch.set_num_threads(max(1, torch.get_num_threads()))
        rng = np.random.default_rng(self.seed)
        d = np.nan_to_num(dev, nan=0.0).astype(np.float32)
        n_cells, _, n_feat = d.shape
        W = self.window

        class AE(nn.Module):
            def __init__(self, n: int, h: int) -> None:
                super().__init__()
                self.enc = nn.LSTM(n, h, batch_first=True)
                self.dec = nn.LSTM(h, h, batch_first=True)
                self.out = nn.Linear(h, n)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                _, (hn, _) = self.enc(x)
                z = hn[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
                y, _ = self.dec(z)
                return self.out(y)

        def windows(cell_idx: np.ndarray, end_idx: np.ndarray) -> np.ndarray:
            offs = np.arange(-W + 1, 1)
            return d[cell_idx[:, None], end_idx[:, None] + offs[None, :]]  # (N, W, F)

        valid_train = train_times[train_times >= W - 1]
        n_all = n_cells * len(valid_train)
        pick = rng.choice(n_all, size=min(self.n_train + 20_000, n_all), replace=False)
        ci, ti = np.divmod(pick, len(valid_train))
        Xall = windows(ci, valid_train[ti])
        n_null = min(20_000, len(Xall) // 4)  # held-out windows -> null error distribution
        Xnull, Xtr = Xall[:n_null], Xall[n_null:]

        model = AE(n_feat, self.hidden)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        loss_fn = nn.MSELoss()
        Xt = torch.from_numpy(Xtr)
        model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(len(Xt))
            total = 0.0
            for b in range(0, len(Xt), self.batch_size):
                xb = Xt[perm[b : b + self.batch_size]]
                opt.zero_grad()
                loss = loss_fn(model(xb), xb)
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
            log.info("lstm-ae epoch %d loss %.4f", epoch + 1, total / len(Xt))

        model.eval()

        def last_step_error(x: np.ndarray) -> np.ndarray:
            errs = []
            with torch.no_grad():
                for b in range(0, len(x), 4096):
                    xb = torch.from_numpy(x[b : b + 4096])
                    errs.append(((model(xb)[:, -1, :] - xb[:, -1, :]) ** 2).numpy())
            return np.concatenate(errs)  # (N, F)

        null = last_step_error(Xnull)
        tails = [TailModel.fit(null[:, k]) for k in range(n_feat)]
        ci = np.repeat(np.arange(n_cells), len(times))
        ti = np.tile(times, n_cells)
        err = last_step_error(windows(ci, ti)).reshape(n_cells, len(times), n_feat)
        out = np.empty_like(err)
        for k in range(n_feat):
            out[:, :, k] = tails[k].surprisal(err[:, :, k])
        return out
