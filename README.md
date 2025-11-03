# 🏏 IPL Win Probability Predictor

**Name:** IPL Win Probability Predictor

**Description:** Predict real‑time IPL win probabilities from current match context using a calibrated scikit‑learn model and Streamlit UI.

## 🚀 Features

- **Real-time predictions** from match context
- **Interactive Streamlit UI** for inputs and results
- **Model selection** over multiple algorithms with cross‑validation
- **Reproducible training** from public IPL datasets

## 🛠️ Tech Stack

- Python, Streamlit, Pandas, NumPy, scikit‑learn
- Model serialization via Pickle

## 📂 Data

- `matches.csv`: Match‑level metadata (winner, city, etc.)
- `deliveries.csv`: Ball‑by‑ball data (runs, wickets, over/ball, teams)

The training script builds second‑innings, per‑ball states to learn the probability of the chasing side winning from that state.

## 📊 Features Used

- `batting_team`, `bowling_team`, `city`
- `runs_left`, `balls_left`, `wickets`
- `total_runs_x` (target), `crr` (current run rate), `rrr` (required run rate)

Categoricals are one‑hot encoded; numerics are passed through in a single scikit‑learn `Pipeline`.

## 🧠 Modeling Approach

The training pipeline evaluates multiple classifiers and automatically selects the most efficient one by cross‑validated accuracy:

- Logistic Regression (baseline, well‑calibrated)
- Random Forest (non‑linear interactions, tabular strength)
- Gradient Boosting (additive trees, competitive on structured data)

Selection uses 5‑fold cross validation on the same preprocessed features. The best mean CV accuracy model is refit on a train split and evaluated on a held‑out test split. The final pipeline is saved as `pipe.pkl` and consumed directly by `app.py`.

To see the chosen model and its accuracy, run the training script; it prints per‑model CV scores and the final hold‑out accuracy.

## What Model We Use and Why

This project does model selection automatically. During training, we evaluate three scikit‑learn classifiers on identical features and preprocessing:

- Logistic Regression
- Random Forest
- Gradient Boosting

We pick the model with the best cross‑validated accuracy and then wrap it in a single `Pipeline` with preprocessing and apply probability calibration using `CalibratedClassifierCV` (isotonic). The result is saved as `pipe.pkl`.

Why this setup:

- Well‑calibrated probabilities: Win probability must be numerically meaningful, not just ranked; isotonic calibration improves probability quality.
- Strong tabular performance: Tree‑based models (Random Forest / Gradient Boosting) capture non‑linear interactions; Logistic Regression provides a strong, simple baseline.
- Robust preprocessing: `OneHotEncoder(handle_unknown='ignore')` makes the model resilient to unseen teams/cities.
- Fast inference: The final calibrated pipeline is lightweight enough for real‑time Streamlit use.
- Transparent and reproducible: Cross‑validated model choice is logged; the app shows the active classifier under “Model info”.

Note: The exact chosen classifier can differ across datasets/runs (depending on data and random seeds). Check the training output or the app’s “Model info” expander to see which classifier is currently deployed and whether calibration is active.

## ▶️ Quick Start

Install dependencies:
```bash
pip install -r requirements.txt
```

Train the model from real data:
```bash
python create_mock_model.py               # full training
# or a faster, lighter run on low-spec machines
python create_mock_model.py --quick       # downsample + lighter CV/params
```

Run the app:
```bash
streamlit run app.py
```

The app runs at `http://localhost:8501`.

### Windows one-liners (CMD and PowerShell)

Command Prompt (CMD):
```bat
cd /d C:\Users\swara\OneDrive\ドキュメント\achine-Learning-powered-IPL-match-win-probability-predictor-with-Streamlit-web-app-main
python create_mock_model.py
streamlit run app.py
```

PowerShell:
```powershell
Set-Location "C:\Users\swara\OneDrive\ドキュメント\achine-Learning-powered-IPL-match-win-probability-predictor-with-Streamlit-web-app-main"
python create_mock_model.py
streamlit run app.py
```

## 📦 Project Structure

```
ipl-win-probability-predictor/
├── app.py                   # Streamlit app
├── create_mock_model.py     # Training script (real data + model selection)
├── matches.csv              # Match metadata
├── deliveries.csv           # Ball-by-ball data
├── pipe.pkl                 # Saved model pipeline
├── requirements.txt         # Dependencies
├── Procfile                 # (Optional) Deployment config
├── setup.sh                 # (Optional) Streamlit config
└── README.md                # Project documentation
```

## 🔍 Reproducible Metrics

The training script reports:
- Per‑model 5‑fold CV accuracy (mean ± std)
- Chosen model name
- Hold‑out test accuracy

Re‑run the script after updating data or parameters to regenerate metrics.

## 📝 License

MIT License.
