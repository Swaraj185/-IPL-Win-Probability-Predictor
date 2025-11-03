import streamlit as st
import pickle
import pandas as pd

teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

import os
model_path = os.path.join(os.path.dirname(__file__), 'pipe.pkl')
pipe = pickle.load(open(model_path,'rb'))
try:
    base_pipe = getattr(pipe, 'estimator', getattr(pipe, 'base_estimator', pipe))
    clf = base_pipe.named_steps.get('clf') if hasattr(base_pipe, 'named_steps') else None
    clf_name = type(clf).__name__ if clf is not None else type(pipe).__name__
    calibrated = hasattr(pipe, 'base_estimator')
except Exception:
    clf_name = 'Unknown'
    calibrated = False
st.title('🏏 IPL Win Probability Predictor')
with st.expander('Model info'):
    st.write(f"Classifier: {clf_name}")
    st.write(f"Calibrated: {'Yes' if calibrated else 'No'}")
st.markdown("**Developed by Swaraj** | *Machine Learning Project*")

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team',sorted(teams))

selected_city = st.selectbox('Select host city',sorted(cities))

st.markdown("### Match Situation")
target = st.number_input('Target (1st innings total + 1)', min_value=1, value=150, help="Enter the target runs (first innings total + 1)")

col3,col4,col5 = st.columns(3)

with col3:
    score = st.number_input('Current Score', min_value=0, value=0)
with col4:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, value=0.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets Out', min_value=0, max_value=10, value=0)

if st.button('Predict Probability'):
    runs_left = max(0, int(target - score))
    balls_bowled = int(round(overs * 6))
    balls_bowled = min(max(balls_bowled, 0), 120)
    balls_left = max(120 - balls_bowled, 1)
    wickets_in_hand = max(0, min(10, 10 - int(wickets)))
    crr = (score / (balls_bowled / 6.0)) if balls_bowled > 0 else 0.0
    rrr = (runs_left * 6.0) / balls_left

    # Wicket-pressure features
    is_tail = 1 if wickets_in_hand <= 2 else 0
    is_last_wicket = 1 if wickets_in_hand == 1 else 0
    wickets_pressure = 10.0 / (wickets_in_hand + 0.5)

    # Additional interaction/pressure features
    pressure = rrr - crr
    runs_per_wicket_left = (runs_left / wickets_in_hand) if wickets_in_hand > 0 else 1e6
    balls_per_wicket_left = (balls_left / wickets_in_hand) if wickets_in_hand > 0 else 1e6
    wicket_weighted_rrr = rrr * (10.0 / (wickets_in_hand + 0.5))

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_in_hand],
        'total_runs_x': [int(target)],
        'crr': [crr],
        'rrr': [rrr],
        'is_tail': [is_tail],
        'is_last_wicket': [is_last_wicket],
        'wickets_pressure': [wickets_pressure],
        'pressure': [pressure],
        'runs_per_wicket_left': [runs_per_wicket_left],
        'balls_per_wicket_left': [balls_per_wicket_left],
        'wicket_weighted_rrr': [wicket_weighted_rrr]
    })

    result = pipe.predict_proba(input_df)
    loss = float(result[0][0])
    win = float(result[0][1])

    # Tail-risk adjustment: apply conservative cap when very few wickets remain
    if wickets_in_hand <= 1:
        risk_factor = 0.35
    elif wickets_in_hand == 2:
        risk_factor = 0.60
    elif wickets_in_hand == 3:
        risk_factor = 0.80
    else:
        risk_factor = 1.00

    # Slightly increase penalty when many balls are left (long time to survive)
    if wickets_in_hand <= 2 and balls_left > 36:
        risk_factor *= 0.9

    win = max(0.0, min(1.0, win * risk_factor))
    loss = 1.0 - win
    st.header(f"🏆 {batting_team}: {round(win*100)}%")
    st.header(f"🎯 {bowling_team}: {round(loss*100)}%")
    
    # Add some personal touches
    st.markdown("---")
    st.markdown("**Prediction by Swaraj's ML Model**")
    st.markdown("*Built with Python, Streamlit & Scikit-learn*")