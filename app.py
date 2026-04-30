import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Social Media Engagement AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>

/* Background */
html, body, [data-testid="stAppViewContainer"] {
    background: #0A1224 !important;
    color: #E6ECFF;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0D1B2A !important;
    border-right: 1px solid #1B2A44;
}
[data-testid="stSidebar"] * {
    color: #C7D2FE !important;
}

/* Header */
.header-banner {
    position: relative;
    padding: 28px 32px;
    border-radius: 18px;
    
    background: radial-gradient(circle at 20% 20%, #1E3A8A, #0F172A 60%);
    
    border: 1px solid rgba(59, 130, 246, 0.25);
    
    box-shadow:
        0 10px 30px rgba(0, 0, 0, 0.6),
        inset 0 0 40px rgba(56, 189, 248, 0.05);
    
    overflow: hidden;
}

/* subtle glow line */
.header-banner::after {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    height: 2px;
    width: 100%;
    background: linear-gradient(90deg, transparent, #38BDF8, transparent);
    opacity: 0.6;
}

/* title */
.header-banner h1 {
    font-size: 34px;
    font-weight: 700;
    margin-bottom: 6px;
    
    background: linear-gradient(90deg, #38BDF8, #60A5FA, #A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* subtitle */
.header-banner p {
    font-size: 15px;
    color: #94A3B8;
    letter-spacing: 0.3px;
}
}
.header-banner {
    background: linear-gradient(
        120deg,
        #0F172A,
        #1E3A8A,
        #2563EB,
        #0F172A
    );
    background-size: 300% 300%;
    animation: gradientMove 8s ease infinite;
}

/* gradient animation */
@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
/* shimmer layer */
.header-banner::before {
    content: "";
    position: absolute;
    top: 0;
    left: -100%;
    width: 50%;
    height: 100%;

    background: linear-gradient(
        120deg,
        transparent,
        rgba(255,255,255,0.08),
        transparent
    );

    transform: skewX(-20deg);
    animation: shimmerMove 6s infinite;
}

/* shimmer animation */
@keyframes shimmerMove {
    0% { left: -100%; }
    100% { left: 150%; }
}
/* Metric cards */
.metric-card {
    position: relative;
    padding: 12px 16px;
    border-radius: 10px;
    background: #0F172A;
    border: 1px solid #1E293B;
}

/* NEW accent bar */
.metric-card::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    height: 100%;
    width: 4px;
    background: #38BDF8;
    border-radius: 10px 0 0 10px;
}

/* Prediction cards */
.pred-result {
    background: linear-gradient(135deg, #022C22, #064E3B);
    border: 1px solid #065F46;
}
.pred-result .big-num {
    color: #34D399;
}

/* Section headers */
.section-header {
    color: #E0E7FF;
    border-left: 3px solid #38BDF8;
}

/* Insight cards */
.insight-card {
    background: #0F172A;
    border: 1px solid #1E293B;
    color: #CBD5F5;
}
.insight-card strong {
    color: #38BDF8;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #2563EB, #38BDF8);
    color: white;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #1D4ED8, #0EA5E9);
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #94A3B8 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #38BDF8 !important;
    border-bottom-color: #38BDF8 !important;
}

/* Dataframe */
.stDataFrame {
    border: 1px solid #1E293B;
}

/* Slider track (filled part) */
[data-baseweb="slider"] div[role="slider"] {
    background-color: #38BDF8 !important;
}

/* Slider background (rail) */
[data-baseweb="slider"] > div > div {
    background-color: #1E293B !important;
}

/* Slider thumb (circle) */
[data-baseweb="slider"] div[role="slider"] > div {
    background-color: #38BDF8 !important;
}

/* Toggle switch (on state) */
[data-baseweb="switch"] div[aria-checked="true"] {
    background-color: #38BDF8 !important;
}
/* Hide slider value label */
[data-baseweb="slider"] div[data-testid="stThumbValue"] {
    display: none !important;
}
/* Reduce spacing between inputs */
[data-testid="stSidebar"] .stNumberInput,
[data-testid="stSidebar"] .stSelectbox,
[data-testid="stSidebar"] .stSlider {
    margin-bottom: 8px !important;
}
/* Compact dropdown */
[data-baseweb="select"] {
    font-size: 14px !important;
}
/* metric card base (add if not already present) */
.metric-card {
    opacity: 0;
    transform: translateY(20px) scale(0.98);
    animation: cardFadeUp 0.6s ease forwards;
}

/* stagger effect */
.metric-card:nth-child(1) { animation-delay: 0.1s; }
.metric-card:nth-child(2) { animation-delay: 0.2s; }
.metric-card:nth-child(3) { animation-delay: 0.3s; }
.metric-card:nth-child(4) { animation-delay: 0.4s; }
.metric-card:nth-child(5) { animation-delay: 0.5s; }

/* animation keyframe */
@keyframes cardFadeUp {
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}
.metric-card {
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.metric-card:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 10px 25px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

# ── Data Generation + Model (cached) ─────────────────────────────────────────
@st.cache_resource(show_spinner="🔄  Training models on synthetic dataset...")
def build_pipeline():
    np.random.seed(42)
    N = 2000

    platforms     = ['Instagram', 'Facebook', 'Twitter']
    content_types = ['Image', 'Video', 'Text', 'Carousel', 'Reel']
    topics        = ['Fashion', 'Food', 'Tech', 'Travel', 'Fitness',
                     'Beauty', 'Finance', 'Gaming', 'News', 'Lifestyle']

    platform_bias  = {'Instagram': 1.3, 'Facebook': 1.0, 'Twitter': 0.8}
    content_bias   = {'Image': 1.1, 'Video': 1.5, 'Text': 0.7, 'Carousel': 1.3, 'Reel': 1.8}
    topic_bias     = {'Fashion': 1.2, 'Food': 1.3, 'Tech': 1.0, 'Travel': 1.4, 'Fitness': 1.1,
                      'Beauty': 1.2, 'Finance': 0.9, 'Gaming': 1.1, 'News': 0.8, 'Lifestyle': 1.0}
    peak_hours     = [8, 9, 12, 17, 18, 19, 20, 21]

    plat  = np.random.choice(platforms,     N)
    ctype = np.random.choice(content_types, N)
    topic = np.random.choice(topics,        N)
    hour  = np.random.randint(0, 24, N)
    dow   = np.random.randint(0, 7,  N)
    followers      = np.random.lognormal(9, 1.5, N).astype(int).clip(100, 5_000_000)
    prev_avg_eng   = np.random.lognormal(5, 1.2, N).clip(10, 50_000)
    num_hashtags   = np.random.randint(0, 31, N)
    caption_length = np.random.randint(10, 2200, N)
    has_cta        = np.random.randint(0, 2, N)
    is_sponsored   = np.random.randint(0, 2, N)

    base = (
        followers * 0.02
        * np.array([platform_bias[p] for p in plat])
        * np.array([content_bias[c]  for c in ctype])
        * np.array([topic_bias[t]    for t in topic])
        * (1 + 0.3 * np.isin(hour, peak_hours).astype(float))
        * (1 + 0.2 * np.isin(dow, [5, 6]).astype(float))
        * (1 + 0.15 * has_cta)
        * (1 + 0.10 * is_sponsored)
        * (1 + 0.3  * (prev_avg_eng / prev_avg_eng.max()))
        * (1 + 0.05 * np.clip(num_hashtags, 0, 15) / 15)
        * np.random.lognormal(0, 0.5, N)
    ).clip(1)

    total_likes    = (base * np.random.uniform(0.4, 0.7, N)).astype(int)
    total_shares   = (base * np.random.uniform(0.05, 0.2, N)).astype(int)
    total_comments = (base * np.random.uniform(0.05, 0.15, N)).astype(int)
    engagement_score = total_likes + total_shares * 2 + total_comments * 1.5

    df = pd.DataFrame({
        'platform': plat, 'content_type': ctype, 'topic': topic,
        'hour_posted': hour, 'day_of_week': dow, 'followers': followers,
        'prev_avg_engagement': prev_avg_eng,
        'num_hashtags': num_hashtags, 'caption_length': caption_length,
        'has_cta': has_cta, 'is_sponsored': is_sponsored,
        'total_likes': total_likes, 'total_shares': total_shares,
        'total_comments': total_comments, 'engagement_score': engagement_score,
    })

    raw_df = pd.DataFrame({
        'platform': plat, 'content_type': ctype, 'topic': topic,
        'hour_posted': hour, 'day_of_week': dow, 'followers': followers,
        'num_hashtags': num_hashtags,
        'is_peak_hour': np.isin(hour, peak_hours).astype(int),
        'is_weekend': (dow >= 5).astype(int),
        'engagement_score': engagement_score,
    })

    # Feature engineering
    df['is_peak_hour']     = df['hour_posted'].isin(peak_hours).astype(int)
    df['is_weekend']       = (df['day_of_week'] >= 5).astype(int)
    df['log_followers']    = np.log1p(df['followers'])
    df['log_prev_eng']     = np.log1p(df['prev_avg_engagement'])
    df['log_eng_score']    = np.log1p(df['engagement_score'])
    df['hashtag_optimal']  = ((df['num_hashtags'] >= 5) & (df['num_hashtags'] <= 15)).astype(int)
    df['caption_short']    = (df['caption_length'] < 150).astype(int)
    df['caption_long']     = (df['caption_length'] > 500).astype(int)
    df['eng_per_follower'] = df['engagement_score'] / (df['followers'] + 1)
    df['hour_sin']         = np.sin(2 * np.pi * df['hour_posted'] / 24)
    df['hour_cos']         = np.cos(2 * np.pi * df['hour_posted'] / 24)
    df['dow_sin']          = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos']          = np.cos(2 * np.pi * df['day_of_week'] / 7)

    df = pd.get_dummies(df, columns=['platform', 'content_type', 'topic'], drop_first=True)

    FEATURES = [
        'log_followers', 'log_prev_eng', 'num_hashtags', 'caption_length',
        'has_cta', 'is_sponsored', 'is_peak_hour', 'is_weekend',
        'hashtag_optimal', 'caption_short', 'caption_long',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    ] + [c for c in df.columns if c.startswith(('platform_', 'content_type_', 'topic_'))]

    X = df[FEATURES]
    y = df['log_eng_score']

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    lr_pipe = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
    lr_cv_rmse = np.sqrt(-cross_val_score(lr_pipe, X, y, cv=kf, scoring='neg_mean_squared_error'))
    lr_cv_r2   = cross_val_score(lr_pipe, X, y, cv=kf, scoring='r2')
    lr_pipe.fit(X, y)
    lr_pred = lr_pipe.predict(X)

    rf_pipe = Pipeline([('model', RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1))])
    rf_cv_rmse = np.sqrt(-cross_val_score(rf_pipe, X, y, cv=kf, scoring='neg_mean_squared_error'))
    rf_cv_r2   = cross_val_score(rf_pipe, X, y, cv=kf, scoring='r2')
    rf_pipe.fit(X, y)
    rf_pred = rf_pipe.predict(X)

    rf_model = rf_pipe.named_steps['model']
    fi = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=False)

    return dict(
        df=df, raw_df=raw_df, X=X, y=y,
        FEATURES=FEATURES,
        lr_pipe=lr_pipe, rf_pipe=rf_pipe,
        lr_pred=lr_pred, rf_pred=rf_pred,
        lr_cv_rmse=lr_cv_rmse, lr_cv_r2=lr_cv_r2,
        rf_cv_rmse=rf_cv_rmse, rf_cv_r2=rf_cv_r2,
        fi=fi, peak_hours=peak_hours,
        platforms=platforms, content_types=content_types, topics=topics,
    )

cache = build_pipeline()
df        = cache['df']
raw_df    = cache['raw_df']
X         = cache['X']
y         = cache['y']
lr_pipe   = cache['lr_pipe']
rf_pipe   = cache['rf_pipe']
lr_pred   = cache['lr_pred']
rf_pred   = cache['rf_pred']
lr_cv_rmse = cache['lr_cv_rmse']
lr_cv_r2   = cache['lr_cv_r2']
rf_cv_rmse = cache['rf_cv_rmse']
rf_cv_r2   = cache['rf_cv_r2']
fi         = cache['fi']
peak_hours = cache['peak_hours']
PLATFORMS     = cache['platforms']
CONTENT_TYPES = cache['content_types']
TOPICS        = cache['topics']

# ── Plot theme helper ─────────────────────────────────────────────────────────
BG, CARD, TEXT = '#0A0A12', '#14142A', '#E0E0F0'
ACC1, ACC2 = "#1427D2", "#1F07BA"
COLORS = ["#00F150","#687C7B","#BACF16",'#96CEB4',"#9A937C",
          "#FF0606",'#98D8C8','#F7B731','#5A67D8','#48BB78']

def dark_fig(w=10, h=5):
    plt.rcParams.update({
        'figure.facecolor': BG, 'axes.facecolor': CARD,
        'axes.edgecolor': '#2A2A4A', 'axes.labelcolor': TEXT,
        'xtick.color': TEXT, 'ytick.color': TEXT,
        'text.color': TEXT, 'grid.color': '#2A2A4A',
        'font.family': 'DejaVu Sans',
    })
    return plt.subplots(figsize=(w, h), facecolor=BG)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📡 Engagement Predictor")
    st.markdown("---")
    st.markdown("*Post Configuration*")

    sb_platform = st.selectbox("Platform", PLATFORMS)
    sb_content  = st.selectbox("Content Type", CONTENT_TYPES)
    sb_topic    = st.selectbox("Topic", TOPICS)
    sb_followers = st.number_input("Followers", min_value=100, max_value=5_000_000,
                                    value=50_000, step=1000)
    sb_prev_eng  = st.number_input("Avg Previous Engagement", min_value=10,
                                    max_value=50_000, value=1200, step=100)
    sb_hour      = st.slider("Hour Posted (0–23)", 0, 23, 18)
    sb_dow       = st.selectbox("Day of Week",
                                ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    sb_hashtags  = st.slider("# Hashtags", 0, 30, 10)
    sb_caption   = st.slider("Caption Length (chars)", 10, 2200, 200)
    sb_cta       = st.toggle("Has Call-to-Action", value=True)
    sb_sponsored = st.toggle("Sponsored Post", value=False)

    st.markdown("---")
    st.markdown("*Model*")
    sb_model = st.radio("Select model", ["Random Forest", "Linear Regression"])

    predict_btn = st.button("🚀 Predict Engagement", use_container_width=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
<h1>Social Media Engagement</h1>
<p>Predict & analyse engagement across platforms using machine learning</p>
</div>
""", unsafe_allow_html=True)
st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
# ── Top Metrics ───────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)

def metric_card(col, label, value, color="#4ECDC4"):
    placeholder = col.empty()

    # small step animation (3 steps max → fast + smooth)
    steps = [0.3, 0.6, 1.0]

    for s in steps:
        display_val = int(float(value) * s) if str(value).isdigit() else value
        placeholder.markdown(f"""
        <div class="metric-card">
          <div class="label">{label}</div>
          <div class="value" style="color:{color}">{display_val}</div>
        </div>
        """, unsafe_allow_html=True)

        import time
        time.sleep(0.08)

metric_card(m1, "Dataset Size", f"{2000:,}", "#4ECDC4")
metric_card(m2, "Features", f"{len(cache['FEATURES']):,}", "#FF6B6B")
metric_card(m3, "RF CV R²", f"{rf_cv_r2.mean():.3f}", "#45B7D1")
metric_card(m4, "RF CV RMSE", f"{rf_cv_rmse.mean():.4f}", "#FFEAA7")
metric_card(m5, "LR CV R²", f"{lr_cv_r2.mean():.3f}", "#96CEB4")

st.markdown("<br>", unsafe_allow_html=True)

# ── Prediction Panel (top if predict clicked) ─────────────────────────────────
if predict_btn:
    dow_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,
               "Friday":4,"Saturday":5,"Sunday":6}
    dow_val = dow_map[sb_dow]

    # Build one-row feature vector matching training columns
    sample_base = X.iloc[0:1].copy() * 0

    # Continuous
    sample_base['log_followers']   = np.log1p(sb_followers)
    sample_base['log_prev_eng']    = np.log1p(sb_prev_eng)
    sample_base['num_hashtags']    = sb_hashtags
    sample_base['caption_length']  = sb_caption
    sample_base['has_cta']         = int(sb_cta)
    sample_base['is_sponsored']    = int(sb_sponsored)
    sample_base['is_peak_hour']    = int(sb_hour in peak_hours)
    sample_base['is_weekend']      = int(dow_val >= 5)
    sample_base['hashtag_optimal'] = int(5 <= sb_hashtags <= 15)
    sample_base['caption_short']   = int(sb_caption < 150)
    sample_base['caption_long']    = int(sb_caption > 500)
    sample_base['hour_sin']        = np.sin(2 * np.pi * sb_hour / 24)
    sample_base['hour_cos']        = np.cos(2 * np.pi * sb_hour / 24)
    sample_base['dow_sin']         = np.sin(2 * np.pi * dow_val / 7)
    sample_base['dow_cos']         = np.cos(2 * np.pi * dow_val / 7)

    # One-hot dummies
    for col in X.columns:
        if col.startswith('platform_'):
            p = col.replace('platform_', '')
            sample_base[col] = int(sb_platform == p)
        elif col.startswith('content_type_'):
            c = col.replace('content_type_', '')
            sample_base[col] = int(sb_content == c)
        elif col.startswith('topic_'):
            t = col.replace('topic_', '')
            sample_base[col] = int(sb_topic == t)

    pipe = rf_pipe if sb_model == "Random Forest" else lr_pipe
    pred_log = pipe.predict(sample_base)[0]
    pred_eng = int(np.expm1(pred_log))
    likes    = int(pred_eng * 0.55)
    shares   = int(pred_eng * 0.12)
    comments = int(pred_eng * 0.10)

    st.markdown('<div class="section-header">🎯 Prediction Result</div>', unsafe_allow_html=True)
    pa, pb, pc, pd_ = st.columns(4)

    def pred_card(col, icon, label, val, color):
        col.markdown(f"""
        <div class="pred-result">
          <div class="label">{icon} {label}</div>
          <div class="big-num" style="color:{color}">{val:,}</div>
        </div>""", unsafe_allow_html=True)

    pred_card(pa, "🔥", "Total Engagement", pred_eng, "#4ECDC4")
    pred_card(pb, "❤️",  "Est. Likes",       likes,    "#FF6B6B")
    pred_card(pc, "🔄", "Est. Shares",      shares,   "#45B7D1")
    pred_card(pd_, "💬", "Est. Comments",    comments, "#96CEB4")

    hour_tip = "✅ Peak hour!" if sb_hour in peak_hours else "⚠️ Off-peak — try 8–9AM or 5–9PM"
    hash_tip = "✅ Optimal range" if 5 <= sb_hashtags <= 15 else "⚠️ Use 5–15 hashtags"
    st.info(f"*Hour Tip:* {hour_tip}  |  *Hashtag Tip:* {hash_tip}")
    st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Analytics", "🤖 Model Performance", "🔍 Feature Importance", "📋 Data Explorer"])

# ════════════════════════ TAB 1 – Analytics ═══════════════════════════════════
with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header"> Engagement by Platform</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(7, 6)
        plat_data = [raw_df[raw_df['platform'] == p]['engagement_score'].values for p in PLATFORMS]
        bp = ax.boxplot(plat_data, labels=PLATFORMS, patch_artist=True,
                        medianprops={'color':'white','lw':2})
        for patch, color in zip(bp['boxes'], COLORS):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_yscale('log'); ax.grid(axis='y', alpha=0.3)
        ax.set_ylabel('Engagement Score')
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        st.markdown('<div class="section-header"> Median Engagement by Content Type</div>', unsafe_allow_html=True)
        ct_avg = raw_df.groupby('content_type')['engagement_score'].median().sort_values()
        fig, ax = dark_fig(6, 4)
        ax.barh(ct_avg.index, ct_avg.values, color=COLORS[:len(ct_avg)], edgecolor='none')
        ax.set_xlabel('Median Engagement'); ax.grid(axis='x', alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-header"> Hourly Engagement Trend</div>', unsafe_allow_html=True)
        hr_med = raw_df.groupby('hour_posted')['engagement_score'].median()
        fig, ax = dark_fig(6, 4)
        ax.plot(hr_med.index, hr_med.values, color=ACC2, lw=2.5, marker='o', ms=4)
        ax.fill_between(hr_med.index, hr_med.values, alpha=0.2, color=ACC2)
        for h in peak_hours:
            ax.axvline(h, color=ACC1, alpha=0.35, lw=1)
        ax.set_xlabel('Hour (0–23)'); ax.set_ylabel('Median Engagement')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    with col_d:
        st.markdown('<div class="section-header"> Engagement vs Hashtag Count</div>', unsafe_allow_html=True)
        ht_bins = pd.cut(raw_df['num_hashtags'], bins=[0,2,5,10,15,20,30], include_lowest=True)
        ht_med  = raw_df.groupby(ht_bins, observed=True)['engagement_score'].median()
        labels  = ['1-2','3-5','6-10','11-15','16-20','21-30']
        fig, ax = dark_fig(6, 4)
        ax.bar(range(len(ht_med)), ht_med.values, color=COLORS[:len(ht_med)], edgecolor='none')
        ax.set_xticks(range(len(ht_med))); ax.set_xticklabels(labels)
        ax.set_xlabel('# Hashtags'); ax.set_ylabel('Median Engagement')
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header"> Content Type × Day of Week Heatmap</div>', unsafe_allow_html=True)
    pivot = raw_df.pivot_table(values='engagement_score', index='content_type',
                               columns='day_of_week', aggfunc='median')
    pivot.columns = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    fig, ax = dark_fig(12, 4)
    sns.heatmap(pivot, ax=ax, cmap='RdYlGn', fmt='.0f', annot=True,
                linewidths=0.5, linecolor='#0A0A12',
                cbar_kws={'label':'Median Engagement'})
    ax.set_xlabel('Day of Week'); ax.set_ylabel('Content Type')
    fig.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header"> Strategy Insights</div>', unsafe_allow_html=True)
    insights = [
        (" Reels & Video win", "Reels carry a <strong>1.8× engagement multiplier</strong> — the highest of any content format. Prioritise short-form video over static images."),
        (" Time your posts", "Post during peak windows: <strong>8–9 AM, 12 PM, 5–9 PM</strong>. These hours show a consistent 30% uplift in median engagement."),
        ("#️ Hashtag sweet spot", "Use <strong>5–15 hashtags</strong>. Under 5 leaves reach on the table; over 15 yields diminishing returns and can look spammy."),
        (" Always add a CTA", "A clear <strong>Call-to-Action</strong> boosts engagement by ~15%. Ask your audience something specific."),
        (" Topic matters", "<strong>Travel & Food</strong> consistently outperform Finance & News. Align content with high-affinity categories where possible."),
        (" Weekend boost", "Saturday & Sunday posts earn <strong>~20% more</strong> engagement than weekdays on average."),
        ("📈 Consistency is king", "Past average engagement is the <strong>single strongest predictor</strong>. Build momentum — a consistent posting cadence compounds."),
    ]
    c1, c2 = st.columns(2)
    for i, (title, body) in enumerate(insights):
        col = c1 if i % 2 == 0 else c2
        col.markdown(f'<div class="insight-card"><strong>{title}</strong><br>{body}</div>',
                     unsafe_allow_html=True)

# ════════════════════════ TAB 2 – Model Performance ═══════════════════════════
with tab2:
    col_e, col_f = st.columns(2)

    with col_e:
        st.markdown('<div class="section-header">📉 5-Fold CV RMSE</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(6, 4)
        models_  = ['Linear\nRegression', 'Random\nForest']
        cv_rmses = [lr_cv_rmse.mean(), rf_cv_rmse.mean()]
        cv_stds  = [lr_cv_rmse.std(),  rf_cv_rmse.std()]
        bars = ax.bar(models_, cv_rmses, color=[ACC1, ACC2], width=0.5, edgecolor='none')
        ax.errorbar(models_, cv_rmses, yerr=cv_stds, fmt='none', color='white', capsize=6, lw=2)
        for bar, val in zip(bars, cv_rmses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=11, color=TEXT)
        ax.set_ylabel('RMSE (log scale)'); ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    with col_f:
        st.markdown('<div class="section-header">📈 CV R² Score</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(6, 4)
        r2s = [lr_cv_r2.mean(), rf_cv_r2.mean()]
        ax.bar(models_, r2s, color=[ACC1, ACC2], width=0.5, edgecolor='none')
        for i, v in enumerate(r2s):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=13, fontweight='bold', color=TEXT)
        ax.set_ylabel('R² Score'); ax.set_ylim(0, 1); ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">🎯 Actual vs Predicted (Log Scale, n=500)</div>', unsafe_allow_html=True)
    fig, ax = dark_fig(12, 5)
    idx = np.random.choice(len(y), 500, replace=False)
    ax.scatter(y.iloc[idx], rf_pred[idx], alpha=0.4, s=15, color=ACC2, label='RF Predicted')
    ax.scatter(y.iloc[idx], lr_pred[idx], alpha=0.3, s=15, color=ACC1, label='LR Predicted')
    lo, hi = y.min(), y.max()
    ax.plot([lo, hi], [lo, hi], 'white', lw=1.5, ls='--', label='Perfect fit')
    ax.set_xlabel('Actual Log Engagement'); ax.set_ylabel('Predicted Log Engagement')
    ax.legend(fontsize=9); ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">📊 RF Residual Distribution</div>', unsafe_allow_html=True)
    fig, ax = dark_fig(12, 3.5)
    rf_resid = y.values - rf_pred
    ax.hist(rf_resid, bins=80, color=ACC2, alpha=0.85, edgecolor='none')
    ax.axvline(0, color='white', lw=2, ls='--')
    ax.set_xlabel('Residual'); ax.set_ylabel('Count'); ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig); plt.close()

    # Summary table
    st.markdown('<div class="section-header">📋 Model Summary</div>', unsafe_allow_html=True)
    summary = pd.DataFrame({
        'Model':    ['Linear Regression', 'Random Forest'],
        'CV RMSE':  [f"{lr_cv_rmse.mean():.4f} ± {lr_cv_rmse.std():.4f}",
                     f"{rf_cv_rmse.mean():.4f} ± {rf_cv_rmse.std():.4f}"],
        'CV R²':    [f"{lr_cv_r2.mean():.4f}", f"{rf_cv_r2.mean():.4f}"],
        'Train RMSE': [f"{np.sqrt(mean_squared_error(y, lr_pred)):.4f}",
                       f"{np.sqrt(mean_squared_error(y, rf_pred)):.4f}"],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

# ════════════════════════ TAB 3 – Feature Importance ══════════════════════════
with tab3:
    st.markdown('<div class="section-header">🌲 Top-20 Feature Importances (Random Forest)</div>', unsafe_allow_html=True)
    top20 = fi.head(20)
    fig, ax = dark_fig(12, 7)
    colors_fi = [ACC1 if i < 5 else ACC2 if i < 10 else '#5A67D8' for i in range(len(top20))]
    top20[::-1].plot(kind='barh', ax=ax, color=colors_fi[::-1], edgecolor='none')
    ax.axvline(top20.mean(), color='white', linestyle='--', alpha=0.5, label='Mean importance')
    ax.set_xlabel('Importance Score'); ax.legend(fontsize=9); ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">📊 Full Feature Importance Table</div>', unsafe_allow_html=True)
    fi_df = fi.reset_index()
    fi_df.columns = ['Feature', 'Importance']
    fi_df['Rank'] = range(1, len(fi_df)+1)
    fi_df = fi_df[['Rank','Feature','Importance']]
    fi_df['Importance'] = fi_df['Importance'].round(5)
    st.dataframe(fi_df, use_container_width=True, hide_index=True, height=400)

# ════════════════════════ TAB 4 – Data Explorer ════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">🗂️ Raw Dataset (first 500 rows)</div>', unsafe_allow_html=True)
    show_cols = ['platform','content_type','topic','hour_posted','day_of_week',
                 'followers','num_hashtags','has_cta','is_sponsored',
                 'total_likes','total_shares','total_comments','engagement_score']
    display_df = raw_df[['platform','content_type','topic','hour_posted','day_of_week',
                          'followers','num_hashtags','is_peak_hour','is_weekend','engagement_score']].copy()
    st.dataframe(display_df.head(500), use_container_width=True, height=400)

    st.markdown('<div class="section-header">📈 Descriptive Statistics</div>', unsafe_allow_html=True)
    num_cols = ['followers','num_hashtags','is_peak_hour','is_weekend','engagement_score']
    st.dataframe(display_df[num_cols].describe().round(2), use_container_width=True)

    st.markdown('<div class="section-header">🏷️ Topic Performance Ranking</div>', unsafe_allow_html=True)
    topic_med = raw_df.groupby('topic')['engagement_score'].agg(['median','mean','std']).sort_values('median', ascending=False).round(0)
    topic_med.columns = ['Median','Mean','Std Dev']
    st.dataframe(topic_med, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#444466; font-size:0.8rem;'>"
    "Social Media Engagement Prediction System · Synthetic dataset · "
    "Random Forest + Linear Regression · 5-Fold Cross-Validation"
    "</p>",
    unsafe_allow_html=True
)
