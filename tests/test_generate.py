import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generate_data import make_facilities, simulate  # noqa: E402


def build(seed):
    rng = np.random.default_rng(seed)
    fac = make_facilities(rng, 30)
    return simulate(rng, fac, 24, "2023-01")


def test_same_seed_gives_identical_data():
    assert build(1).equals(build(1))


def test_different_seed_gives_different_data():
    assert not build(1).equals(build(2))


def test_no_negative_stock_and_flags_consistent():
    df = build(3).dropna(subset=["stock_on_hand"])
    assert (df["stock_on_hand"] >= 0).all()
    assert ((df["stockout_days"] > 0).astype(int) == df["stockout_flag"]).all()


def test_remote_facilities_have_more_stockouts():
    df = build(4).dropna(subset=["stockout_flag"])
    rate = df.groupby("remoteness")["stockout_flag"].mean()
    assert rate["remote"] > rate["urban"]
