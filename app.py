import streamlit as st
import pandas as pd
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
import plotly.express as px
# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Intelligent Player Valuation System",
    page_icon="⚽",
    layout="wide"
)

# ================= CSS =================
st.markdown("""
<style>

/* ===== IMPORT FONTS ===== */
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=EB+Garamond:wght@400;500;600&display=swap');

/* ===== BACKGROUND ===== */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: #f8fafc;
}

/* ===== GLOBAL FONT ===== */
html, body, [class*="css"] {
    font-family: 'EB Garamond', serif;
}

/* ===== CENTER ===== */
.center-container {
    text-align: center;
    margin-top: 50px;
}

/* ===== TITLE ===== */
.main-title {
    font-family: 'Sora', sans-serif;
    font-size: 56px;
    font-weight: 700;
    color: #f8fafc;
}

/* ===== SUBTITLE ===== */
.subtitle {
    font-size: 18px;
    color: #e8d5c4;
    margin-top: 10px;
}

/* ===== BUTTON ===== */
.stButton > button {
    background: rgba(255,255,255,0.05);
    color: #f8fafc;
    border: 1px solid rgba(255,255,255,0.2);
    padding: 12px;
    border-radius: 10px;
    font-weight: 600;
    transition: 0.3s;
}

.stButton > button:hover {
    background: rgba(232,213,196,0.15);
    border: 1px solid #e8d5c4;
    box-shadow: 0px 0px 12px rgba(232,213,196,0.5);
    transform: scale(1.02);
}

/* ===== GLASS CARD ===== */
.glass {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.3);
}

/* ===== PLAYER NAME GLOW ===== */
.player-name {
    font-size: 28px;
    font-weight: 600;
    color: #f8fafc;
    text-shadow: 0px 0px 10px rgba(232,213,196,0.4);
    transition: 0.3s;
}

.player-name:hover {
    text-shadow: 0px 0px 18px rgba(232,213,196,0.8);
}

/* ===== PLAYER CARD ===== */
.player-card {
    background: rgba(255,255,255,0.06);
    padding: 18px;
    border-radius: 14px;
    text-align: center;
    transition: 0.3s;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.25);
}

.player-card:hover {
    transform: translateY(-6px) scale(1.04);
    box-shadow: 0px 15px 30px rgba(0,0,0,0.45);
}

/* ===== METRIC HOVER WHITE ===== */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 12px;
    transition: 0.3s;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-6px);
    box-shadow: 0px 10px 25px rgba(255,255,255,0.2);
}

/* ===== EFFICIENCY CARD ===== */
.eff-card {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    transition: 0.3s;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.25);
}

.eff-card.green:hover {
    transform: translateY(-6px);
    box-shadow: 0px 0px 20px rgba(74,222,128,0.8);
    color: #4ade80;
}

.eff-card.red:hover {
    transform: translateY(-6px);
    box-shadow: 0px 0px 20px rgba(248,113,113,0.8);
    color: #f87171;
}

.eff-card.white:hover {
    transform: translateY(-6px);
    box-shadow: 0px 0px 22px rgba(255,255,255,0.85);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD DATA =================
df = pd.read_csv("players_20.csv")

df = df.dropna(subset=["short_name","age","overall","potential","value_eur","player_positions"])

# ================= ML =================
X = df[["age","overall","potential"]]
y = df["value_eur"] / 1_000_000

rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
rf_model.fit(X, y)

# ================= PLAYER LIST =================
players = sorted(df["short_name"].unique())

# ================= HEADER =================
st.markdown("""
<div class="center-container">
    <div class="main-title">⚽ Intelligent Player Valuation System</div>
    <div class="subtitle">AI-based Player Valuation & Similarity Engine</div>
</div>
""", unsafe_allow_html=True)

# ===== DESCRIPTION =====
st.markdown("""
<div style="
    text-align: center;
    max-width: 720px;
    margin: 25px auto 40px auto;
    font-size: 18px;
    line-height: 1.7;
    color: #cbd5e1;
">
    This system leverages machine learning to estimate a football player’s market value 
    while identifying comparable players based on performance attributes. 
    It combines predictive modeling with similarity analysis to support smarter scouting decisions.
</div>
""", unsafe_allow_html=True)

# ================= INPUT =================
col1, col2, col3 = st.columns([1,2,1])

with col2:
    player = st.selectbox("🔍 Search Player", players)
    predict = st.button("🚀 Run Model", use_container_width=True)

# ================= RESULTS =================
if predict:

    # ===== PREMIUM LOADING =====
    progress_text = st.empty()
    progress_bar = st.progress(0)

    steps = [
        "📂 Loading player data...",
        "🧠 Running valuation model...",
        "🎯 Finding similar players...",
        "📊 Building dashboard...",
        "✅ Finalizing results..."
    ]

    for i, step in enumerate(steps):
        progress_text.markdown(
            f"<div style='text-align:center;color:#e8d5c4;font-size:18px;'>{step}</div>",
            unsafe_allow_html=True
        )
        progress_bar.progress((i + 1) * 20)
        time.sleep(0.28)

    progress_text.empty()
    progress_bar.empty()

    # ===== PLAYER DATA =====
    player_data = df[df["short_name"] == player].iloc[0]

    input_data = [[
        player_data["age"],
        player_data["overall"],
        player_data["potential"]
    ]]

    predicted_value = rf_model.predict(input_data)[0]
    value_m = round(predicted_value, 2)

    # ===== EFFICIENCY =====
    actual_value = player_data["value_eur"] / 1_000_000
    efficiency = round(predicted_value / max(actual_value, 0.1), 2)

    if efficiency > 1.15:
        label = "~U"
        eff_class = "green"
        eff_color = "#4ade80"

    elif efficiency < 0.85:
        label = "~O"
        eff_class = "red"
        eff_color = "#f87171"

    else:
        label = "~F"
        eff_class = "white"
        eff_color = "#ffffff"
        
    # ===== HERO =====
    st.markdown(f"""
    <div class="glass" style="text-align:center;">
        <h2 class="player-name">{player}</h2>
        <h5 style="color:#e8d5c4;">{player_data['player_positions']}</h5>
    </div>
    """, unsafe_allow_html=True)

    # ===== DESCRIPTION =====
    desc = f"""
    A {player_data['player_positions']} aged {player_data['age']} with an overall rating of {player_data['overall']}
    and potential of {player_data['potential']}. This player shows
    {"high growth potential" if player_data['potential'] - player_data['overall'] > 5 else "consistent performance"}
    and is currently valued at €{round(player_data['value_eur']/1_000_000,2)}M.
    """

    st.markdown(f"""
    <div class="glass" style="text-align:center; padding:18px;">
        <div style="color:#cbd5e1; max-width:760px; margin:auto; font-size:20px; line-height:1.8;">
            {desc}
        
    </div>
    """, unsafe_allow_html=True)

    # ===== METRICS =====
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("💰 Value (€M)", value_m)
    col2.metric("📈 Overall", player_data["overall"])
    col3.metric("🔮 Potential", player_data["potential"])

    col4.markdown(f"""
    <div class="eff-card {eff_class}">
        <h5>⚖️ Efficiency</h5>
        <h2>{efficiency} {label}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ===== SIMILAR PLAYERS =====
    st.markdown("<h2 style='text-align:center;'>Similar Players</h2>", unsafe_allow_html=True)

    main_position = player_data["player_positions"].split(",")[0]
    filtered_df = df[df["player_positions"].str.contains(main_position)]

    knn_model = NearestNeighbors(n_neighbors=6)
    knn_model.fit(filtered_df[["age","overall","potential"]])

    _, indices = knn_model.kneighbors(input_data)
    similar_players = filtered_df.iloc[indices[0][1:5]]

    cols = st.columns(4)

    for i, (_, row) in enumerate(similar_players.iterrows()):
        with cols[i]:
            st.markdown(f"""
            <div class="player-card">
                <h4>{row['short_name']}</h4>
                <p style="color:#e8d5c4;">{row['player_positions']}</p>
                <p>⭐ {row['overall']}</p>
            </div>
            """, unsafe_allow_html=True)
    # ===== DASHBOARD HEADER =====
    st.markdown("""
    <div style="text-align:center; margin-top:10px; margin-bottom:18px;">
    <h1 style="
    font-family:Sora,sans-serif;
    font-size:46px;
    color:#f8fafc;
    margin-bottom:8px;">
    Player Analytics Dashboard
    </h1>

    <div style="
    color:#e8d5c4;
    font-size:18px;">
    AI Comparison • Similarity Intelligence • Market Insights
    </div>
    </div>
    """, unsafe_allow_html=True)

    # ===== GRAPHS =====
    g1, g2 = st.columns(2)

    compare = pd.concat([
        pd.DataFrame([player_data]),
        similar_players.head(3)
    ])

    with g1:
        fig = go.Figure()

        cats = ["overall", "potential", "age"]

        for _, row in compare.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[
                    row["overall"],
                    row["potential"],
                    row["age"]
                ],
                theta=["Overall", "Potential", "Age"],
                fill='toself',
                name=row["short_name"]
            ))

        fig.update_layout(
            polar=dict(bgcolor="#0f172a"),
            paper_bgcolor="#0f172a",
            font_color="white",
            height=500
        )

        st.markdown('<div class="graph-box">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with g2:
        scatter = px.scatter(
            compare,
            x="overall",
            y="value_eur",
            text="short_name",
            size="potential",
            color="age"
        )

        scatter.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font_color="white",
            height=500
        )

        st.markdown('<div class="graph-box">', unsafe_allow_html=True)
        st.plotly_chart(scatter, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)