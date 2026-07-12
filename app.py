import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Set page config for a premium wide layout
st.set_page_config(
    page_title="xG Tactician — Expected Goals Simulator",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
    }
    .metric-value-large {
        font-size: 3.5rem;
        font-weight: 800;
        color: #fbbf24;
        line-height: 1;
        text-shadow: 0 0 10px rgba(251, 191, 38, 0.2);
    }
    .stProgress > div > div > div > div {
        background-color: #fbbf24;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #334155;
        border-radius: 0.5rem;
        background-color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

# Add api directory to sys.path to load geometry features
sys.path.append(os.path.join(os.path.dirname(__file__), 'api'))
try:
    from api.geometry import (
        dist, shot_angle, freeze_frame_features, 
        GOAL_X, GOAL_Y_CENTER, POST_LEFT, POST_RIGHT, point_in_triangle
    )
except ImportError:
    # Fallback to local import if sys.path addition needs help
    from api.geometry import (
        dist, shot_angle, freeze_frame_features, 
        GOAL_X, GOAL_Y_CENTER, POST_LEFT, POST_RIGHT, point_in_triangle
    )

# --- CACHED MODEL LOADERS ---
@st.cache_resource
def load_xg_model():
    """Loads the trained production Logistic Regression pipeline."""
    return joblib.load(os.path.join("api", "xg_model.joblib"))

@st.cache_data
def load_metrics():
    """Loads the training/testing metrics."""
    with open(os.path.join("models", "metrics.json")) as f:
        return joblib.load(os.path.join("models", "metrics.json")) if hasattr(joblib, 'load') else pd.read_json(f, typ='series')

# We'll parse metrics using json instead to be robust
import json
with open(os.path.join("models", "metrics.json")) as f:
    MODEL_INFO = json.load(f)

@st.cache_resource
def get_baseline_model():
    """Trains a simple distance + angle baseline model on the fly from shots.csv."""
    try:
        df = pd.read_csv(os.path.join('data', 'shots.csv'))
        TEST_TAG = '55_282'  # Euro 2024 held out
        train_df = df[df['tournament'] != TEST_TAG].reset_index(drop=True)
        
        baseline_pipe = Pipeline([
            ('prep', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000)),
        ])
        baseline_pipe.fit(train_df[['distance_to_goal', 'shot_angle_rad']], train_df['is_goal'])
        return baseline_pipe
    except Exception as e:
        st.warning(f"Could not load data to train baseline model on-the-fly: {e}")
        return None

# Load models
model = load_xg_model()
baseline_model = get_baseline_model()

# --- APP LAYOUT ---
st.title("⚽ xG Tactician")
st.markdown("*Expected Goals (xG) Simulator & Defensive Context Explainer*")

# Create tabs for Simulator vs Model Details
tab_sim, tab_model = st.tabs(["🎮 Interactive Simulator", "📊 Model Analysis & Calibration"])

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("🎯 Shot Qualifiers")
    
    st.subheader("Shot Coordinates")
    shot_x = st.slider("Shot X (distance from own goal)", 60.0, 119.5, 108.5, step=0.5, 
                       help="Goal line is at x=120. Attacking half starts at x=60.")
    shot_y = st.slider("Shot Y (width, 0-80)", 5.0, 75.0, 42.0, step=0.5,
                       help="Goal center is at y=40. Goal mouth is between y=36 and y=44.")
    
    st.subheader("Shot Context")
    body_part = st.selectbox("Body Part", ['Right Foot', 'Left Foot', 'Head', 'Other'], index=2)
    technique = st.selectbox("Technique", ['Normal', 'Volley', 'Lob', 'Half Volley', 'Overhead Kick', 'Diving Header', 'Backheel'], index=0)
    shot_type = st.selectbox("Shot Type", ['Open Play', 'Free Kick', 'Corner', 'Kick Off'], index=0)
    play_pattern = st.selectbox("Play Pattern", [
        'Regular Play', 'From Corner', 'From Throw In', 'From Goal Kick', 
        'From Keeper', 'From Kick Off', 'From Counter', 'From Free Kick', 'Other'
    ], index=0)
    
    st.subheader("Flags")
    col1, col2 = st.columns(2)
    with col1:
        under_pressure = st.checkbox("Under Pressure")
        first_time = st.checkbox("First Time")
        open_goal = st.checkbox("Open Goal")
    with col2:
        deflected = st.checkbox("Deflected")
        aerial_won = st.checkbox("Aerial Duel Won")
        follows_dribble = st.checkbox("Follows Dribble")
        
    st.subheader("Game State")
    minute = st.slider("Game Minute", 0, 120, 45)
    score_diff_before_shot = st.slider("Goal Difference", -5, 5, 0, 
                                       help="Shooting team's goal difference. +1 means shooting team is leading by 1.")

# --- TAB 1: INTERACTIVE SIMULATOR ---
with tab_sim:
    # Layout splits into visualization & statistics
    col_pitch, col_stats = st.columns([7, 5])
    
    with col_stats:
        st.subheader("🔮 Predictions & Impact")
        
        # We define a default opponent list for user editing
        if 'opponents_df' not in st.session_state:
            st.session_state.opponents_df = pd.DataFrame([
                {"x": 116.0, "y": 40.0, "Role": "Goalkeeper"},
                {"x": 112.0, "y": 38.5, "Role": "Defender"},
                {"x": 115.0, "y": 43.0, "Role": "Defender"}
            ])
            
        st.markdown("**🛡️ Position Defensive Players**")
        st.caption("Double-click cells to change positions. Use the '+' button to add players, or select a row and press Delete.")
        
        edited_df = st.data_editor(
            st.session_state.opponents_df,
            num_rows="dynamic",
            column_config={
                "x": st.column_config.NumberColumn("X Coord (60-120)", min_value=60.0, max_value=120.0, step=0.5, format="%.1f"),
                "y": st.column_config.NumberColumn("Y Coord (0-80)", min_value=0.0, max_value=80.0, step=0.5, format="%.1f"),
                "Role": st.column_config.SelectboxColumn("Role", options=["Defender", "Goalkeeper"], required=True),
            },
            width="stretch",
            key="opponents_editor"
        )
        
        # Parse edited opponents
        opponents = []
        for _, r in edited_df.iterrows():
            opponents.append({
                'x': float(r['x']),
                'y': float(r['y']),
                'position_name': r['Role']
            })
            
        # Run geometry calculations
        distance = dist(shot_x, shot_y, GOAL_X, GOAL_Y_CENTER)
        angle = shot_angle(shot_x, shot_y)
        ff_feats = freeze_frame_features(shot_x, shot_y, opponents)
        
        # Prepare inputs
        inputs = {
            "distance_to_goal": distance,
            "shot_angle_rad": angle,
            "minute": minute,
            "score_diff_before_shot": score_diff_before_shot,
            "n_opponents_in_cone": ff_feats["n_opponents_in_cone"],
            "n_opponents_total": ff_feats["n_opponents_total"],
            "nearest_opponent_dist": ff_feats["nearest_opponent_dist"],
            "gk_dist_to_goal_center": ff_feats["gk_dist_to_goal_center"],
            "gk_dist_from_shot_line": ff_feats["gk_dist_from_shot_line"],
            "body_part": body_part,
            "technique": technique,
            "shot_type": shot_type,
            "play_pattern": play_pattern,
            "under_pressure": under_pressure,
            "first_time": first_time,
            "open_goal": open_goal,
            "deflected": deflected,
            "aerial_won": aerial_won,
            "follows_dribble": follows_dribble,
        }
        
        # Run prediction
        X_df = pd.DataFrame([inputs])
        xg = float(model.predict_proba(X_df)[0, 1])
        
        # Run baseline prediction if available
        baseline_xg = 0.0
        if baseline_model is not None:
            baseline_xg = float(baseline_model.predict_proba(pd.DataFrame([{"distance_to_goal": distance, "shot_angle_rad": angle}]))[0, 1])
            
        # Display large prediction metric
        st.markdown(
            f'<div style="text-align: center; margin-top: 1rem;">'
            f'<span style="font-size: 1.1rem; color: #94a3b8; font-weight:600;">EXPECTED GOALS (xG)</span><br>'
            f'<span class="metric-value-large">{xg:.3f}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.progress(xg)
        
        # Metric comparison layout
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Model xG probability", f"{xg*100:.1f}%")
        with m_col2:
            if baseline_model is not None:
                delta = xg - baseline_xg
                st.metric("Baseline xG (dist+angle)", f"{baseline_xg*100:.1f}%", delta=f"{delta*100:+.1f}%")
            else:
                st.metric("Baseline xG", "N/A")
                
        # Explain prediction via logs-odds
        try:
            prep = model.named_steps['prep']
            clf = model.named_steps['clf']
            X_trans = prep.transform(X_df)
            if hasattr(X_trans, 'toarray'):
                X_trans = X_trans.toarray()
            feature_names = prep.get_feature_names_out()
            coefs = clf.coef_[0]
            
            contributions = X_trans[0] * coefs
            
            groups = {
                'Distance & Angle': 0.0,
                'Defensive Context': 0.0,
                'Shot Quality & Flags': 0.0,
                'Body Part & Technique': 0.0,
                'Game State (Min/Score)': 0.0
            }
            
            for name, contrib in zip(feature_names, contributions):
                if 'distance_to_goal' in name or 'shot_angle_rad' in name:
                    groups['Distance & Angle'] += contrib
                elif any(k in name for k in ['opponents', 'gk_', 'nearest_opponent']):
                    groups['Defensive Context'] += contrib
                elif any(k in name for k in ['under_pressure', 'first_time', 'open_goal', 'deflected', 'aerial_won', 'follows_dribble']):
                    groups['Shot Quality & Flags'] += contrib
                elif any(k in name for k in ['body_part', 'technique', 'shot_type', 'play_pattern']):
                    groups['Body Part & Technique'] += contrib
                elif 'minute' in name or 'score_diff_before_shot' in name:
                    groups['Game State (Min/Score)'] += contrib
                    
            contrib_df = pd.DataFrame([
                {"Factor": k, "Impact (Log-Odds)": v} for k, v in groups.items()
            ]).sort_values("Impact (Log-Odds)", ascending=True)
            
            # Plotly horizontal bar chart
            fig_contrib = go.Figure()
            fig_contrib.add_trace(go.Bar(
                y=contrib_df["Factor"],
                x=contrib_df["Impact (Log-Odds)"],
                orientation='h',
                marker=dict(
                    color=['#ef4444' if x < 0 else '#10b981' for x in contrib_df["Impact (Log-Odds)"]],
                    line=dict(color='rgba(0,0,0,0)', width=0)
                ),
                text=contrib_df["Impact (Log-Odds)"].apply(lambda val: f"{val:+.2f}"),
                textposition='inside',
                insidetextanchor='middle'
            ))
            fig_contrib.update_layout(
                title="<b>Log-Odds Feature Breakdown</b>",
                xaxis_title="Negative Impact (Red) vs Positive Impact (Green)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f8fafc', size=11),
                margin=dict(l=10, r=10, t=40, b=10),
                height=250,
                xaxis=dict(showgrid=True, gridcolor='#334155', zeroline=True, zerolinecolor='#64748b')
            )
            st.plotly_chart(fig_contrib, use_container_width=True, config={'displayModeBar': False})
        except Exception as ex:
            st.error(f"Could not compute feature impact breakdown: {ex}")
            
    with col_pitch:
        st.subheader("🏟️ Tactical Board")
        
        # Build soccer pitch lines
        pitch_traces = []
        
        # 1. Boundary line (x >= 60 to 120, y 0 to 80)
        pitch_traces.append(go.Scatter(
            x=[60, 120, 120, 60, 60],
            y=[0, 0, 80, 80, 0],
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.4)', width=2.5),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # 2. Penalty box (x 102 to 120, y 18 to 62)
        pitch_traces.append(go.Scatter(
            x=[120, 102, 102, 120],
            y=[18, 18, 62, 62],
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.4)', width=2),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # 3. 6-Yard Box (x 114 to 120, y 30 to 50)
        pitch_traces.append(go.Scatter(
            x=[120, 114, 114, 120],
            y=[30, 30, 50, 50],
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.4)', width=2),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # 4. Penalty Spot (108, 40)
        pitch_traces.append(go.Scatter(
            x=[108],
            y=[40],
            mode='markers',
            marker=dict(color='rgba(255, 255, 255, 0.6)', size=6),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # 5. Penalty Arc (centered at 108, 40 with radius 10, extending x < 102)
        theta_arc = np.linspace(np.arccos(-0.6), 2 * np.pi - np.arccos(-0.6), 50)
        arc_x = 108 + 10 * np.cos(theta_arc)
        arc_y = 40 + 10 * np.sin(theta_arc)
        pitch_traces.append(go.Scatter(
            x=arc_x,
            y=arc_y,
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.4)', width=2),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # 6. Center Circle (semi-circle at x=60, radius 10)
        theta_circle = np.linspace(-np.pi/2, np.pi/2, 50)
        circle_x = 60 + 10 * np.cos(theta_circle)
        circle_y = 40 + 10 * np.sin(theta_circle)
        pitch_traces.append(go.Scatter(
            x=circle_x,
            y=circle_y,
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.4)', width=2),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # 7. Goal mouth net (outside the field, x 120-122, y 36-44)
        pitch_traces.append(go.Scatter(
            x=[120, 122, 122, 120],
            y=[36, 36, 44, 44],
            mode='lines',
            fill='toself',
            fillcolor='rgba(255, 255, 255, 0.05)',
            line=dict(color='rgba(255, 255, 255, 0.3)', width=1.5),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # 8. Shot Cone (triangle from shot to left post (120, 36) and right post (120, 44))
        pitch_traces.append(go.Scatter(
            x=[shot_x, 120, 120, shot_x],
            y=[shot_y, 36, 44, shot_y],
            mode='lines',
            fill='toself',
            fillcolor='rgba(251, 191, 36, 0.12)',
            line=dict(color='rgba(251, 191, 36, 0.4)', width=1.5, dash='dash'),
            name='Shot Cone',
            text=f'Angle: {angle * 57.29578:.1f}°',
            hoverinfo='name+text'
        ))
        
        # 9. Line from shot to center of goalmouth (120, 40)
        pitch_traces.append(go.Scatter(
            x=[shot_x, 120],
            y=[shot_y, 40],
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.25)', width=1.5, dash='dot'),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # 10. Goalkeeper projections & connections (perpendicular distance to shot line)
        gk_x, gk_y = [], []
        def_in_cone_x, def_in_cone_y = [], []
        def_out_cone_x, def_out_cone_y = [], []
        
        for opp in opponents:
            if opp['position_name'] == 'Goalkeeper':
                gk_x.append(opp['x'])
                gk_y.append(opp['y'])
                
                # Perpendicular connection to shot line
                sx, sy = shot_x, shot_y
                gx, gy = opp['x'], opp['y']
                vx, vy = 120.0 - sx, 40.0 - sy
                wx, wy = gx - sx, gy - sy
                len_sq = vx * vx + vy * vy
                if len_sq > 0:
                    t = max(0.0, min(1.0, (wx * vx + wy * vy) / len_sq))
                    proj_x = sx + t * vx
                    proj_y = sy + t * vy
                    
                    pitch_traces.append(go.Scatter(
                        x=[gx, proj_x],
                        y=[gy, proj_y],
                        mode='lines',
                        line=dict(color='rgba(6, 182, 212, 0.6)', width=1.5, dash='dash'),
                        name='GK Positioning',
                        text=f'Perp. Dist: {ff_feats["gk_dist_from_shot_line"]:.1f}y',
                        hoverinfo='name+text',
                        showlegend=False
                    ))
            else:
                # Test if defender is in shot cone
                in_cone = point_in_triangle(opp['x'], opp['y'], shot_x, shot_y, 120, 36, 120, 44)
                if in_cone:
                    def_in_cone_x.append(opp['x'])
                    def_in_cone_y.append(opp['y'])
                else:
                    def_out_cone_x.append(opp['x'])
                    def_out_cone_y.append(opp['y'])
                    
        # Plot Goalkeeper
        if gk_x:
            pitch_traces.append(go.Scatter(
                x=gk_x,
                y=gk_y,
                mode='markers',
                marker=dict(color='#06b6d4', size=13, line=dict(color='#ffffff', width=2), symbol='circle'),
                name='Goalkeeper (GK)',
                text=[f"GK ({x:.1f}, {y:.1f})" for x, y in zip(gk_x, gk_y)],
                hoverinfo='text'
            ))
            
        # Plot Defenders inside Shot Cone
        if def_in_cone_x:
            pitch_traces.append(go.Scatter(
                x=def_in_cone_x,
                y=def_in_cone_y,
                mode='markers',
                marker=dict(color='#ef4444', size=11, line=dict(color='#ffe4e6', width=2.5), symbol='circle'),
                name='Defender (in cone)',
                text=[f"Defender ({x:.1f}, {y:.1f})" for x, y in zip(def_in_cone_x, def_in_cone_y)],
                hoverinfo='text'
            ))
            
        # Plot Defenders outside Shot Cone
        if def_out_cone_x:
            pitch_traces.append(go.Scatter(
                x=def_out_cone_x,
                y=def_out_cone_y,
                mode='markers',
                marker=dict(color='#dc2626', size=10, line=dict(color='#ffffff', width=1), symbol='circle'),
                name='Defender (out of cone)',
                text=[f"Defender ({x:.1f}, {y:.1f})" for x, y in zip(def_out_cone_x, def_out_cone_y)],
                hoverinfo='text'
            ))
            
        # Plot Shot Location (Star)
        pitch_traces.append(go.Scatter(
            x=[shot_x],
            y=[shot_y],
            mode='markers',
            marker=dict(color='#fbbf24', size=16, symbol='star', line=dict(color='#ffffff', width=2)),
            name='Shot Location',
            text=f"Shot Coordinates: ({shot_x:.1f}, {shot_y:.1f})",
            hoverinfo='text'
        ))
        
        # Build Figure
        fig_pitch = go.Figure(data=pitch_traces)
        
        # Layout customizations (tactical board look)
        fig_pitch.update_layout(
            xaxis=dict(
                range=[58, 124], 
                showgrid=False, 
                zeroline=False, 
                showticklabels=False, 
                fixedrange=True
            ),
            yaxis=dict(
                range=[-2, 82], 
                showgrid=False, 
                zeroline=False, 
                showticklabels=False, 
                scaleanchor="x", 
                scaleratio=1,
                fixedrange=True
            ),
            plot_bgcolor='#111827',  # dark navy pitch area
            paper_bgcolor='#0e1117', # dark card background
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                font=dict(color='#f8fafc', size=10)
            ),
            height=460,
        )
        
        st.plotly_chart(fig_pitch, use_container_width=True, config={'displayModeBar': False})
        
        # Display Geometric Feature Summary
        st.markdown("### 🔍 Computed Features & Geometry")
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            st.metric("Distance to Goal", f"{distance:.1f} yards")
            st.metric("Shot Angle", f"{angle * 57.29578:.1f}°")
        with f_col2:
            st.metric("Defenders in Cone", f"{ff_feats['n_opponents_in_cone'] if ff_feats['n_opponents_in_cone'] is not None else 0}")
            st.metric("Nearest Defender", f"{ff_feats['nearest_opponent_dist']:.1f} yards" if ff_feats['nearest_opponent_dist'] is not None else "N/A")
        with f_col3:
            st.metric("GK Distance to Center", f"{ff_feats['gk_dist_to_goal_center']:.1f} yards" if ff_feats['gk_dist_to_goal_center'] is not None else "N/A")
            st.metric("GK Offset from Shot Line", f"{ff_feats['gk_dist_from_shot_line']:.1f} yards" if ff_feats['gk_dist_from_shot_line'] is not None else "N/A")

# --- TAB 2: MODEL DETAILS ---
with tab_model:
    st.subheader("📈 Model Comparison and Generalization")
    
    st.markdown("""
    This Expected Goals (xG) model is evaluated on an **entire held-out tournament (UEFA Euro 2024)** that the model never saw during training or tuning. 
    This mimics a true production deployment where the model is tested on unseen competitions with different playing styles and conversion rates.
    """)
    
    # Render metrics table
    res = MODEL_INFO["results"]
    perf_data = []
    for model_name, metrics in res.items():
        perf_data.append({
            "Model Name": model_name,
            "ROC-AUC (Higher is better)": f"{metrics['auc']:.4f}",
            "Log Loss (Lower is better)": f"{metrics['log_loss']:.4f}",
            "Brier Score (Lower is better)": f"{metrics['brier']:.4f}"
        })
    st.table(pd.DataFrame(perf_data))
    
    col_img, col_desc = st.columns([6, 6])
    
    with col_img:
        # Load and display precomputed model evaluation charts
        eval_img_path = os.path.join("evaluation", "model_evaluation.png")
        if os.path.exists(eval_img_path):
            st.image(eval_img_path, caption="ROC-AUC Curve and Calibration curves across all 4 benchmarked configurations.", use_container_width=True)
        else:
            st.warning("Evaluation charts (model_evaluation.png) not found.")
            
    with col_desc:
        st.markdown("""
        ### 🧠 Logistic Regression vs XGBoost Comparison
        
        #### Why Logistic Regression was shipped:
        - **Regularized Logistic Regression** (AUC = **0.794**) matched or slightly exceeded both **StatsBomb's proprietary model** (AUC = 0.787) and a tuned **XGBoost classifier** (AUC = 0.774) on the Euro 2024 test tournament.
        - Because we have already pre-engineered the non-linear spatial geometry (distance to goal center and the angle subtended by the goal mouth) in the training phase, the main advantage of tree-based models (automatically discovering non-linear coordinate splits) is reduced.
        - With a modest dataset (~6.7k training rows), Logistic Regression exhibits lower variance and generalizes better to held-out tournaments.
        
        #### Key xG Insights (Logistic Regression Coefficients):
        - **Deflection** is the single strongest positive multiplier (deflected shots significantly wrong-foot goalkeepers).
        - **Distance** is the strongest negative driver, decay is exponential.
        - **Open Goal** and a wider **Shot Angle** strongly push xG up.
        - **Headers** and **Overhead Kicks** have strong negative coefficients compared to Foot strikes (controlling for location).
        """)
