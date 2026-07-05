# Claim Denial Risk Predictor — GUI

A minimal Flask app that wraps your existing `model.ipynb` pipeline in a
web form: enter claim/appointment details, get back a denial probability
and risk category (Low / Medium / High), using the exact same cutoffs
your notebook computes.

## How it fits together

```
model.ipynb (yours)  --export_model.py-->  model.pkl + feature_schema.json
                                                    |
                                                    v
                                            app.py (Flask API)
                                                    |
                                                    v
                                       templates/index.html (form + result)
```

The GUI never hardcodes your feature list. It asks the backend for the
schema (`/api/features`), which was generated straight from your training
data, so if you retrain with different columns the form updates itself —
no HTML editing needed.

## 1. Export the trained model from your notebook

Open `model.ipynb` in Colab/Jupyter and run it as usual through **Section
10 ("Risk score function B")** so `final_model`, `X`, `numeric_features`,
`categorical_features`, `low_cutoff`, and `high_cutoff` exist in memory.

Then paste the contents of `export_model.py` (in this folder) into a new
cell at the end and run it. It will write two files into your Colab
working directory:

- `model.pkl`
- `feature_schema.json`

Download both (in Colab: the Files panel → right-click → Download, or
`from google.colab import files; files.download("model.pkl")`).

## 2. Set up this project

Copy `model.pkl` and `feature_schema.json` into this same folder, next to
`app.py`, so the layout looks like:

```
claim_denial_app/
  app.py
  model.pkl              <- copied from Colab
  feature_schema.json    <- copied from Colab
  requirements.txt
  templates/
    index.html
```

Install dependencies (a virtual environment is recommended):

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> If your best model turned out to be XGBoost, `xgboost` must be
> installed for `joblib.load` to unpickle it — it's already in
> `requirements.txt`.

## 3. Run the app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser. The form fields are
generated automatically from `feature_schema.json`:

- Numeric columns become number inputs (with the training data's min/max
  shown as a hint).
- Categorical columns with 30 or fewer distinct values become dropdowns
  populated with the values seen in training; higher-cardinality columns
  (e.g. free-text codes) become plain text inputs.

Fill in the fields and click **Predict** to see the denial probability,
risk badge, and a probability bar.

## Notes / next steps

- This intentionally has no authentication, database, or deployment
  config yet — it's meant to prove out the backend-to-model wiring first,
  per your request. Before this touches real patient data anywhere
  outside your own machine, it will need proper access control and
  HTTPS, since it's health-record-adjacent data.
- `app.py` returns a clear JSON error (not a raw stack trace) if a
  submitted value doesn't match what the pipeline expects — e.g. a typo
  in a numeric field.
- If a field should really be a fixed dropdown (e.g. a diagnosis code
  list) but training data didn't contain enough variety, you can hand-edit
  the list under `"categorical"` in `feature_schema.json` any time — the
  app re-reads that file at each restart.
- Right now every prediction is a single, one-off submission. If you
  want to batch-score a CSV of claims through the same running server,
  that's a small addition (`/api/predict_batch`) — say the word if you'd
  like that added next.
