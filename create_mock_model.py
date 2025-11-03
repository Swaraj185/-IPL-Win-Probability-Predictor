import os
import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


def load_data(matches_path: str = 'matches.csv', deliveries_path: str = 'deliveries.csv') -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(matches_path):
        raise FileNotFoundError(f"Missing file: {matches_path}")
    if not os.path.exists(deliveries_path):
        raise FileNotFoundError(f"Missing file: {deliveries_path}")
    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)
    return matches, deliveries


def build_training_frame(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    deliveries = deliveries.copy()

    # Standardize city column name (older datasets use 'city' or 'venue')
    if 'city' not in matches.columns:
        if 'venue' in matches.columns:
            matches['city'] = matches['venue']
        else:
            matches['city'] = 'Unknown'

    # First innings totals per match
    first_innings = deliveries[deliveries['inning'] == 1]
    first_totals = first_innings.groupby('match_id')['total_runs'].sum().rename('first_innings_total')

    # Second innings ball-by-ball cumulative stats
    second = deliveries[deliveries['inning'] == 2].copy()
    # Ball index within innings: over (1..20), ball (1..6)
    second['balls_bowled'] = (second['over'] - 1) * 6 + second['ball']
    second.sort_values(['match_id', 'over', 'ball'], inplace=True)
    second['runs_cum'] = second.groupby('match_id')['total_runs'].cumsum()

    # Dismissals in second innings so far
    dismissal_flag = second['player_dismissed'].notna().astype(int)
    second['wickets_down'] = dismissal_flag.groupby(second['match_id']).cumsum()

    # Target per match (first innings total + 1)
    second = second.merge(first_totals, left_on='match_id', right_index=True, how='left')
    second['target'] = second['first_innings_total'] + 1

    # Derive features expected by app.py
    second['runs_left'] = second['target'] - second['runs_cum']
    second['balls_left'] = 120 - second['balls_bowled']
    second['wickets'] = 10 - second['wickets_down']
    # Additional wicket pressure features to strengthen effect of low wickets
    second['is_tail'] = (second['wickets'] <= 2).astype(int)
    second['is_last_wicket'] = (second['wickets'] == 1).astype(int)
    second['wickets_pressure'] = 10.0 / (second['wickets'] + 0.5)
    # Avoid divide by zero for CRR and RRR
    with np.errstate(divide='ignore', invalid='ignore'):
        second['overs_faced'] = second['balls_bowled'] / 6.0
        second['crr'] = np.where(second['overs_faced'] > 0, second['runs_cum'] / second['overs_faced'], 0.0)
        second['rrr'] = np.where(second['balls_left'] > 0, (second['runs_left'] * 6) / second['balls_left'], np.nan)
        # clip extreme rates to reasonable bounds to avoid outliers
        second['crr'] = np.clip(second['crr'], 0, 36.0)
        second['rrr'] = np.clip(second['rrr'], 0, 36.0)

    # Interaction/pressure features derivable from UI state (after crr/rrr are available)
    with np.errstate(divide='ignore', invalid='ignore'):
        second['pressure'] = second['rrr'] - second['crr']
        second['runs_per_wicket_left'] = np.where(second['wickets'] > 0, second['runs_left'] / second['wickets'], np.inf)
        second['balls_per_wicket_left'] = np.where(second['wickets'] > 0, second['balls_left'] / second['wickets'], np.inf)
        second['wicket_weighted_rrr'] = second['rrr'] * (10.0 / (second['wickets'] + 0.5))

    # Attach match meta: city and match winner to label outcome
    meta_cols = ['id', 'city', 'winner']
    # Commonly matches.csv uses 'id' as match id
    if 'id' not in matches.columns:
        # Some datasets use 'match_id' instead
        if 'match_id' in matches.columns:
            matches = matches.rename(columns={'match_id': 'id'})
        else:
            matches['id'] = matches.index
    meta = matches[meta_cols]
    second = second.merge(meta, left_on='match_id', right_on='id', how='left')

    # Clean team name inconsistencies between matches and deliveries if any
    # (Keep as-is; OneHotEncoder(handle_unknown='ignore') will robustly handle unseen labels)

    # Label: did chasing team win?
    # For second innings, batting_team is the chasing team on every ball
    second['result'] = (second['batting_team'] == second['winner']).astype(int)

    # Filter valid rows
    model_df = second[
        (second['balls_left'] > 0) &
        (second['runs_left'] >= 0) &
        second['rrr'].notna()
    ].copy()

    # Align to feature names expected by app.py
    model_df.rename(columns={'target': 'total_runs_x'}, inplace=True)

    features = [
        'batting_team', 'bowling_team', 'city',
        'runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr',
        'is_tail', 'is_last_wicket', 'wickets_pressure',
        'pressure', 'runs_per_wicket_left', 'balls_per_wicket_left', 'wicket_weighted_rrr'
    ]
    keep = features + ['result']
    model_df = model_df[keep]

    # Replace inf with NaN then drop, to satisfy sklearn
    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()
    return model_df


def train_and_save(df: pd.DataFrame, model_path: str = 'pipe.pkl') -> float:
    X = df[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr', 'is_tail', 'is_last_wicket', 'wickets_pressure', 'pressure', 'runs_per_wicket_left', 'balls_per_wicket_left', 'wicket_weighted_rrr']]
    y = df['result']

    trf = ColumnTransformer([
        ('team_city_ohe', OneHotEncoder(drop='first', handle_unknown='ignore'), ['batting_team', 'bowling_team', 'city'])
    ], remainder='passthrough')
    # Define candidate classifiers
    candidates = {
        'logreg': LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'),
        'rf': RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42),
        'gb': GradientBoostingClassifier(random_state=42)
    }

    # Cross-validate each wrapped in the same preprocessing pipeline
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = {}
    for name, clf in candidates.items():
        pipe = Pipeline(steps=[('prep', trf), ('clf', clf)])
        cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=1)
        scores[name] = (cv_scores.mean(), cv_scores.std())

    # Pick best by mean accuracy
    best_name = max(scores.items(), key=lambda kv: kv[1][0])[0]
    best_clf = candidates[best_name]

    # Final train-test split for a held-out score
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    # Fit with probability calibration for better probability quality
    base_pipe = Pipeline(steps=[('prep', trf), ('clf', best_clf)])
    # sklearn >=1.6 uses 'estimator', older uses 'base_estimator'
    try:
        calib = CalibratedClassifierCV(estimator=base_pipe, method='isotonic', cv=3)
    except TypeError:
        calib = CalibratedClassifierCV(base_estimator=base_pipe, method='isotonic', cv=3)
    calib.fit(X_train, y_train)
    acc = calib.score(X_test, y_test)

    with open(model_path, 'wb') as f:
        pickle.dump(calib, f)

    # Print model selection summary
    print("Model selection (5-fold CV accuracy mean ± std):")
    for name, (m, s) in scores.items():
        print(f" - {name}: {m:.4f} ± {s:.4f}")
    print(f"Chosen model: {best_name} | Hold-out accuracy: {acc:.4f}")

    return acc


def main():
    parser = argparse.ArgumentParser(description='Train IPL win probability model')
    parser.add_argument('--quick', action='store_true', help='Faster training: fewer rows, fewer trees, 2-fold CV')
    parser.add_argument('--max_rows', type=int, default=int(os.getenv('MAX_ROWS', '150000')), help='Cap training rows for speed')
    args = parser.parse_args()
    try:
        matches, deliveries = load_data()
    except FileNotFoundError as e:
        print(str(e))
        print("Please place 'matches.csv' and 'deliveries.csv' in the project root.")
        return

    print("Preparing training data from ball-by-ball records...")
    df = build_training_frame(matches, deliveries)
    if df.empty:
        print("No valid training rows produced. Check CSV contents.")
        return

    # Optional downsampling to speed up CV on constrained machines
    cap = 80_000 if args.quick else args.max_rows
    if len(df) > cap:
        df = df.sample(n=cap, random_state=42)
        print(f"Downsampled to {cap:,} rows for faster training")
    print(f"Training rows: {len(df):,}")
    accuracy = train_and_save(df, 'pipe.pkl')
    print("🏏 IPL Win Probability Model trained from real data")
    print("📦 Saved model as pipe.pkl")
    print(f"📊 Test accuracy: {accuracy:.3f}")
    print("✅ Ready for deployment!")


if __name__ == '__main__':
    main()
