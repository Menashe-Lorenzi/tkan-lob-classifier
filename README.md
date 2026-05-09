# TKAN-LOB Classifier

This project implements an ungated recurrent KAN cell (a gateless variant of the original TKAN of Genet & Inzirillo, 2024)

> Temporal Kolmogorov–Arnold Network for high-frequency Order Flow
> Imbalance classification on LOBSTER tick data, benchmarked against
> a recurrent LSTM baseline.

UCL MSc Computational Finance — independent project. Accompanies the
write-up [`TKAN for High-Frequency OFI Classification.pdf`](./TKAN%20for%20High-Frequency%20OFI%20Classification.pdf).

The full analysis lives in
[`TKAN_LSTM_LOB_Classifier.ipynb`](./TKAN_LSTM_LOB_Classifier.ipynb);
`src/` exposes the loader, feature engineering, models, and the
day-forward training loop the notebook calls into.

<!--
TODO: insert your headline numbers from the notebook, e.g.:

## Headline result

On a one-month panel of the Magnificent-7, TKAN matches the LSTM
baseline on Macro-F1 (TKAN 0.XXX ± 0.YYY vs LSTM 0.XXX ± 0.YYY across
N day-forward folds) while exposing learned per-feature B-spline
activations that the LSTM cannot.
-->

## Method

1. **Raw data.** LOBSTER tick-level limit-order-book snapshots for
   the Magnificent-7 (AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA),
   period 2025-09-22 to 2025-10-20, ten levels per side.
2. **Feature engineering** (`src/features.py`).
   Tick-level Cont OFI is computed from L1 bid/ask price-and-size
   events, then resampled to fixed bars. On top of the resampled
   panel, stationary features are engineered: distance-to-mid at L1
   and L2, multi-level volume imbalance, micro-price pressure,
   rolling 60-bar OFI momentum, and rolling 60-bar mid-return
   volatility. Absolute price columns are dropped before training to
   avoid non-stationary inputs.
3. **Target.** Three-class next-bar OFI direction (down / flat / up)
   defined by the 20% and 80% quantiles of `next_ofi`, computed
   strictly on the training window of each fold.
4. **Models** (`src/models/`).
   - **LSTM baseline** — single-layer recurrent network with a
     last-step linear classification head.
   - **TKAN** — recurrent architecture whose cell replaces the LSTM
     gate stack with a Kolmogorov–Arnold layer based on B-spline
     basis functions (Cox–de Boor recursion implemented from scratch
     in PyTorch over a static padded knot grid on `[-1, 1]`), with a
     SiLU residual base path for training stability.
5. **Validation** (`src/train.py`).
   Day-forward expanding-window CV: train on days `[1..i]`, test
   strictly on day `i+1`. `LOBSequenceDataset` filters sliding
   windows to those whose first and last timestamps fall on the same
   trading date — this is what prevents overnight bleed across the
   close/open gap.
6. **Training.** AdamW with weight decay and gradient clipping,
   class-weighted cross-entropy with label smoothing to counter the
   20/60/20 class split, early stopping on validation loss.
7. **Reporting.** Per-fold and aggregate Macro-F1 / accuracy, learning
   curves, and a B-spline visualisation of the trained KAN layer
   (`src/visualizations.py:visualize_bspline_kan`).

## Repository layout

```
LOB-TKAN/
├── TKAN_LSTM_LOB_Classifier.ipynb        — assembled analysis notebook
├── TKAN for High-Frequency OFI Classification.pdf  — academic write-up
├── src/
│   ├── data_loader.py                    — LOBSTER CSV → orderbook DataFrame
│   ├── features.py                       — Cont OFI + stationary features
│   ├── train.py                          — day-forward CV + training loop
│   ├── visualizations.py                 — intraday plots, fold comparison, B-spline plot
│   └── models/
│       ├── lstm.py                       — baseline LSTM classifier
│       └── tkan_class.py                 — KANLinear + TKANCell + TKANClassifier
├── saved_models/                         — best LSTM/TKAN weights
├── figures/                              — generated plots
├── data/                                 — see "Data" below; not tracked in git
└── requirements.txt
```

## Reproducing

```bash
# 1. Environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Data — see "Data" section below

# 3. Open the notebook and run top to bottom
jupyter lab TKAN_LSTM_LOB_Classifier.ipynb
```

The notebook drives the full pipeline; `src/` modules are imported
from it.

## Data

LOBSTER tick data is **not redistributed in this repository** (≈ 2.5 GB
and academic-license restricted). To reproduce the pipeline:

1. Request a sample from <https://lobsterdata.com/> — free academic
   samples are available, paid for full panels.
2. Place the unzipped CSVs as:

   ```
   data/M7_LOB/<TICKER>/<TICKER>_<YYYY-MM-DD>_34200000_57600000_message_2.csv
   data/M7_LOB/<TICKER>/<TICKER>_<YYYY-MM-DD>_34200000_57600000_orderbook_2.csv
   ```

3. `load_lob_data_range(ticker, start_date, end_date)` in
   `src/data_loader.py` walks this layout automatically.

### LOBSTER schema (for reference)
- **Message file:** `time, type, order_id, size, price, direction`
- **Orderbook file:** ten levels per side, columns
  `ask_price_1, ask_size_1, bid_price_1, bid_size_1, …`

## Implementation notes

- **`KANLinear`** implements the B-spline basis path (Cox–de Boor
  recursion) plus a SiLU residual base path. Inputs are squashed via
  `tanh` before the spline path to keep them inside the grid; spline
  coefficients and base weights are Xavier-initialised.
- **`TKANCell`** concatenates `[x_t, h_{t-1}]` and feeds the result
  through a `KANLinear` layer; the output is `tanh`-squashed for
  recurrence stability.
- **`LOBSequenceDataset`** filters sliding-window indices to those
  whose first and last timestamps fall on the same trading date — the
  guard against overnight signal bleed.
- Class weights `[3.0, 1.0, 3.0]` correspond to the 20/60/20 quantile
  cut. Replace with inverse-frequency if you change the target
  definition.
- `StandardScaler` is refit on each train fold, never on the test
  fold — important for honest walk-forward AUC/F1.

## References

- Cont, R., Kukanov, A. & Stoikov, S. (2014). The price impact of
  order book events. *Journal of Financial Econometrics*.
- Liu, Z. *et al.* (2024). KAN: Kolmogorov–Arnold Networks.
  *arXiv:2404.19756*.
- Genet, R. & Inzirillo, H. (2024). TKAN: Temporal Kolmogorov–Arnold
  Networks. *arXiv:2405.07344*.
- LOBSTER data: <https://lobsterdata.com/>

## License

MIT — see [LICENSE](./LICENSE) (add one if not yet present).
