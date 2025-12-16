"""
Train a *very* stripped-down yield-spread model that uses only:
 • the last trade’s yield-spread (last_ys)
 • reference-data numeric + binary fields
 • reference-data categorical fields
Label stays `new_ys`.
"""

import os, sys
import pickle, pathlib, json
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split
from datetime import datetime

PROJECT_ROOT = "/Users/gil/git/ficc_python"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from automated_training.auxiliary_variables import (
    NON_CAT_FEATURES, BINARY, CATEGORICAL_FEATURES,
    BATCH_SIZE, NUM_EPOCHS
)
from automated_training.auxiliary_functions import (
    fit_encoders, create_summary_of_results
)
from automated_training.set_random_seed import set_seed
set_seed()

from yield_last_ys_model import yield_last_ys_model

EXCLUDE_COLS = [
    "max_ys_ys","max_ys_ago","max_ys_qdiff",
    "min_ys_ys","min_ys_ago","min_ys_qdiff",
    "max_qty_ys","max_qty_ago","max_qty_qdiff",
    "min_ago_ys","min_ago_ago","min_ago_qdiff",
    "D_min_ago_ys","D_min_ago_ago","D_min_ago_qdiff",
    "P_min_ago_ys","P_min_ago_ago","P_min_ago_qdiff",
    "S_min_ago_ys","S_min_ago_ago","S_min_ago_qdiff",
    "max_ys_ttypes","min_ys_ttypes","max_qty_ttypes",
    "min_ago_ttypes","D_min_ago_ttypes",
    "P_min_ago_ttypes","S_min_ago_ttypes",  "max_ys_ttypes",
    "min_ys_ttypes",
    "max_qty_ttypes",
    "min_ago_ttypes",
    "D_min_ago_ttypes",
    "P_min_ago_ttypes",
    "S_min_ago_ttypes"
]

##############################################################################
# 0. config ­­­­ edit only the pkl path for each experiment
##############################################################################
PKL_PATH = "/Users/gil/git/ficc_python/files/data_auxiliary_views_v2_2025-06-24_2025-02-01.pkl"
MODEL_OUT  = "simple_ref_lastys_model.keras"
ENCODERS_OUT = "simple_ref_lastys_encoders.pkl"

##############################################################################
# 1. load data → choose features → label
##############################################################################
df : pd.DataFrame = pd.read_pickle(PKL_PATH) #[:100000]
print(f"{len(df):,} trades loaded from {PKL_PATH}")

# derive last_ys if missing
if "last_ys" not in df.columns and "trade_history" in df.columns:
    df["last_ys"] = df["trade_history"].apply(
        lambda th: th[0,0] if isinstance(th, np.ndarray) and th.size else np.nan
    )

NUMERIC_BINARY   = [c for c in NON_CAT_FEATURES + BINARY          if c not in EXCLUDE_COLS]
CATEGORICAL_USED = [c for c in CATEGORICAL_FEATURES               if c not in EXCLUDE_COLS]
feature_cols     = ["last_ys"] + NUMERIC_BINARY + CATEGORICAL_USED


label_col     = "new_ys"                      # same as the full model

missing = set(feature_cols+[label_col]) - set(df.columns)
assert not missing, f"Dataset missing columns: {missing}"

# optional: quick eyeball of inputs
print(json.dumps(feature_cols, indent=2))

##############################################################################
# 2. fit & cache categorical encoders
##############################################################################
MODEL_NAME = "yield_spread"    

encoders, fmax = fit_encoders(df, CATEGORICAL_USED, MODEL_NAME)

with open(ENCODERS_OUT, "wb") as f: pickle.dump((encoders, fmax), f)
print(f"Saved encoders → {ENCODERS_OUT}")

##############################################################################
# 3. build X/Y tensors in the **same order** the simple model expects
##############################################################################
def build_inputs(data: pd.DataFrame):
    x_list = []
    # last YS
    x_list.append(data["last_ys"].to_numpy(dtype="float32").reshape(-1,1))

    # numeric + binary
    nb = [np.expand_dims(data[col].astype("float32").to_numpy(),1) for col in NUMERIC_BINARY]
    x_list.append(np.concatenate(nb, axis=-1))

    # categoricals (already label-encoded)
    for col in CATEGORICAL_USED:
        x_list.append(encoders[col].transform(data[col]).astype("float32").reshape(-1,1))
    return x_list

X_all = build_inputs(df)
y_all = df[label_col].astype("float32").to_numpy()

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
X_train, y_train = build_inputs(train_df), train_df[label_col].to_numpy("float32")
X_test , y_test  = build_inputs(test_df) , test_df[label_col].to_numpy("float32")

##############################################################################
# 4. model → compile → train
##############################################################################
model = yield_last_ys_model(
    X_train,
    categorical_features=CATEGORICAL_USED,
    non_cat_features=NUMERIC_BINARY, 
    binary_features=[],
    fmax=fmax,
)
model.compile(optimizer="adam", loss="mae", metrics=["mae"])
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)

##############################################################################
# 5. evaluation (pretty MAE table identical to current model emails)
##############################################################################
print("\n===  Summary on hold-out  ===")
create_summary_of_results(model, test_df, X_test, y_test)

##############################################################################
# 6. save artefacts
##############################################################################
model.save(MODEL_OUT)
print(f"Saved model → {MODEL_OUT}")
