import numpy as np


class StateInputTrajectory:
    """Generic state-input trajectory."""
    def __init__(self, ts, xs, us):
        assert len(ts) == len(xs) == len(us)

        self.ts = ts
        self.xs = xs
        self.us = us

    @classmethod
    def load(cls, filename):
        with np.load(filename) as data:
            ts = data["ts"]
            xs = data["xs"]
            us = data["us"]
        return cls(ts=ts, xs=xs, us=us)

    def save(self, filename):
        np.savez_compressed(filename, ts=self.ts, xs=self.xs, us=self.us)

    def __getitem__(self, idx):
        return self.ts[idx], self.xs[idx], self.us[idx]

    def __len__(self):
        return len(self.ts)
