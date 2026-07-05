# ============================================================
# 13. Export model + feature schema for the web app
# ============================================================
# Add this cell to the END of your notebook (model.ipynb), after
# section 10 ("Risk score function B") has run, so that `final_model`,
# `X`, `numeric_features`, `categorical_features`, `low_cutoff` and
# `high_cutoff` all exist in memory.
#
# It writes two files that the Flask app reads at startup:
#   - model.pkl            the fitted sklearn Pipeline (preprocessing + model)
#   - feature_schema.json  the input fields the GUI needs to render,
#                           plus the risk cutoffs, so the app never has
#                           to guess at your data.

import joblib
import json

# --- 1. Save the fitted pipeline -----------------------------------------
joblib.dump(final_model, "model.pkl")

# --- 2. Build a schema describing every input field -----------------------
feature_schema = {"numeric": {}, "categorical": {}}

for col in numeric_features:
    col_data = X[col].dropna()
    feature_schema["numeric"][col] = {
        "min": float(col_data.min()) if len(col_data) else None,
        "max": float(col_data.max()) if len(col_data) else None,
        "mean": round(float(col_data.mean()), 2) if len(col_data) else None,
    }

for col in categorical_features:
    options = sorted(str(v) for v in X[col].dropna().unique().tolist())
    feature_schema["categorical"][col] = options

# --- 3. Save the risk cutoffs so the app matches the notebook exactly -----
risk_cutoffs = {
    "low_cutoff": float(low_cutoff),
    "high_cutoff": float(high_cutoff),
}

schema_out = {
    "model_name": best_model_name,
    "features": feature_schema,
    "risk_cutoffs": risk_cutoffs,
}

with open("feature_schema.json", "w") as f:
    json.dump(schema_out, f, indent=2)

print("Saved model.pkl and feature_schema.json")
print(f"Numeric features ({len(numeric_features)}):", numeric_features)
print(f"Categorical features ({len(categorical_features)}):", categorical_features)
