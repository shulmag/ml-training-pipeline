"""
simple_ref_lastys_inference.py
A notebook-friendly wrapper that:
  • locates the server code tree automatically
  • pulls ref-data & trade-history via individual_pricing.get_data(...)
  • predicts spread with the small model
  • converts spread → $-price with pricing_functions.price_from_ys(...)
"""

import sys, pathlib, pickle, numpy as np, pandas as pd, tensorflow as tf
from datetime import datetime
from pytz import timezone
import pandas as pd


import sys, pathlib,os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/Users/gil/git/ficc/creds.json"

HERE = pathlib.Path(__file__).resolve()

# Find repo root (directory that has BOTH “ficc” and “ficc_python”)
for p in HERE.parents:
    if (p / "ficc").exists() and (p / "ficc_python").exists():
        REPO_ROOT = p
        break
else:
    raise RuntimeError("Could not locate repo root that contains ficc/ and ficc_python/")

SERVER_ROOT = REPO_ROOT / "ficc" / "app_engine" / "demo" / "server"
PYTHON_ROOT = REPO_ROOT / "ficc_python"

sys.path[:0] = [str(SERVER_ROOT), str(PYTHON_ROOT)]   # prepend both


# ─────────── internal imports (now resolvable) ──────────────────────────
from automated_training.auxiliary_variables import NON_CAT_FEATURES, BINARY, CATEGORICAL_FEATURES
from modules import pricing_functions
from modules.individual_pricing import get_data          # <– single-CUSIP wrapper
from modules.ficc.utils import yc_data, nelson_siegel_model
from modules.auxiliary_functions import get_settlement_date
from modules.ficc.utils.nelson_siegel_model import load_model_parameters
from modules.ficc.utils.nelson_siegel_model import get_ficc_ycl_for_target_trade

# ─────────── model artefacts (next to this script) ──────────────────────
MODEL_PATH = HERE.with_name("simple_ref_lastys_model.keras")
ENC_PATH   = HERE.with_name("simple_ref_lastys_encoders.pkl")

model = tf.keras.models.load_model(MODEL_PATH)
encoders, _ = pickle.load(open(ENC_PATH, "rb"))

EXCLUDE = {  # same 26 columns you skipped in training
    "max_ys_ys","max_ys_ago","max_ys_qdiff","min_ys_ys","min_ys_ago","min_ys_qdiff",
    "max_qty_ys","max_qty_ago","max_qty_qdiff","min_ago_ys","min_ago_ago","min_ago_qdiff",
    "D_min_ago_ys","D_min_ago_ago","D_min_ago_qdiff","P_min_ago_ys","P_min_ago_ago",
    "P_min_ago_qdiff","S_min_ago_ys","S_min_ago_ago","S_min_ago_qdiff",
    "max_ys_ttypes","min_ys_ttypes","max_qty_ttypes","min_ago_ttypes",
    "D_min_ago_ttypes","P_min_ago_ttypes","S_min_ago_ttypes",
}

NUMERIC_BINARY = [c for c in NON_CAT_FEATURES + BINARY if c not in EXCLUDE]
CATEGORICAL_USED = [c for c in CATEGORICAL_FEATURES if c not in EXCLUDE]

# ─────────── small helpers ──────────────────────────────────────────────
def last_ys(th):
    return float(th[0, 0]) if isinstance(th, (np.ndarray, list)) and len(th) else np.nan

def prep_inputs(ref, lys):
    xs = [np.array([[lys]], dtype="float32")]

    # numeric + binary   ← 1×N, not N×1
    nb_vec = np.array([[ref[c] for c in NUMERIC_BINARY]], dtype="float32")
    xs.append(nb_vec)

    for col in CATEGORICAL_USED:                 # filtered list
        idx = encoders[col].transform([ref[col]])[0]
        xs.append(np.array([[idx]], dtype="float32"))
    return xs

def predict(cusip: str, qty_k: int, side: str):
    now = datetime.now()
    current_datetime = now
    current_date     = pd.Timestamp(now.date())
    settlement_date  = get_settlement_date(current_date.date())

    df = get_data(
        cusip=cusip,
        quantity=qty_k * 1000,
        current_date=current_date,
        current_datetime=current_datetime,
        settlement_date=settlement_date,
        trade_type=side,
    )

    ref = df.iloc[0]
    ys = float(model.predict(prep_inputs(ref, last_ys(ref["trade_history"])), verbose=0)[0, 0])
    print(f"Model prediction: {ys}")

    # get current curve level
    ficc_ycl = get_ficc_ycl_for_target_trade(ref, current_datetime)
    print(f"ficc ycl: {ficc_ycl} (---)")

    ytw = ficc_ycl + ys
    print(f"Final YTW: {ytw} (---)")

    row = ref.copy()
    row["ficc_ytw"] = ytw/10000           # expected by pricing helper

    price, calc_date = pricing_functions.get_trade_price_from_yield_spread_model(row)

    return {
        "cusip": cusip,
        "trade_type": side,
        "predicted_price": price,
        "predicted_ys": ys,
        "predicted_ytw": ytw,
        "calc_date": calc_date.strftime("%Y-%m-%d") if calc_date else None,
        "quantity": qty_k,
        "settlement_date": settlement_date,
        "ficc_ycl": ficc_ycl,
    }


# quick CLI check
if __name__ == "__main__":
    print(predict("646039YM3", 1000, "D"))
    print(predict("646039YM3", 1000, "S"))
    print(predict("13063D7Q5", 1000, "P"))
    print(predict("13063D7Q5", 1000, "D"))
