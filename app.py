import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="NFL Contract Analyzer",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Page background ── */
    .stApp { background: #f7f6f3; }
    section[data-testid="stSidebar"] { background: #1a1a1a; }
    section[data-testid="stSidebar"] * { color: #e8e6e0 !important; }
    section[data-testid="stSidebar"] .stSlider > div > div { background: #333 !important; }

    /* ── Header ── */
    .nfl-header {
        font-family: 'DM Serif Display', serif;
        font-size: 2.8rem;
        color: #0d0d0d;
        letter-spacing: -0.02em;
        line-height: 1;
        margin-bottom: 0.3rem;
    }
    .nfl-sub {
        font-size: 0.85rem;
        color: #666;
        font-weight: 400;
        letter-spacing: 0.01em;
        max-width: 700px;
        line-height: 1.5;
    }
    .nfl-rule {
        border: none;
        border-top: 2px solid #0d0d0d;
        margin: 1.2rem 0 1.8rem 0;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid #0d0d0d;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding: 0 24px;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #888;
        background: transparent;
        border: none;
        border-radius: 0;
    }
    .stTabs [aria-selected="true"] {
        color: #0d0d0d !important;
        background: #0d0d0d !important;
        color: #f7f6f3 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #e0ddd6;
        border-radius: 4px;
        padding: 1rem 1.2rem;
    }
    [data-testid="metric-container"] label {
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: #888 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1.8rem !important;
        font-weight: 500 !important;
        color: #0d0d0d !important;
    }

    /* ── Verdict badges ── */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 2px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .badge-under { background: #e8f5e9; color: #1b5e20; border: 1px solid #1b5e20; }
    .badge-over  { background: #fdecea; color: #b71c1c; border: 1px solid #b71c1c; }
    .badge-fair  { background: #e8eaf6; color: #1a237e; border: 1px solid #1a237e; }

    /* ── Verdict card ── */
    .verdict-card {
        border-radius: 4px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        border-left: 4px solid;
    }
    .verdict-under { background: #f1f8f1; border-color: #1b5e20; }
    .verdict-over  { background: #fdf4f4; border-color: #b71c1c; }
    .verdict-fair  { background: #f0f2ff; border-color: #1a237e; }
    .verdict-label {
        font-family: 'DM Serif Display', serif;
        font-size: 1.6rem;
        margin-bottom: 0.2rem;
    }
    .verdict-sub { font-size: 0.85rem; color: #555; }

    /* ── Section headers ── */
    .section-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #e0ddd6;
    }

    /* ── Dataframe ── */
    div[data-testid="stDataFrame"] {
        border: 1px solid #e0ddd6;
        border-radius: 4px;
        overflow: hidden;
    }

    /* ── Sidebar labels ── */
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label,
    section[data-testid="stSidebar"] .stSlider label {
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: #aaa !important;
    }

    /* ── Divider ── */
    hr { border-color: #e0ddd6; }

    /* ── Caption ── */
    .stCaption { color: #999 !important; font-size: 0.75rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
UNDERPAID_MIN = 0.5
OVERPAID_MAX  = -0.7
OVERPAID_EXCL = {'RB'}
FUTURE_VALUE_EXCEPTIONS = {"Ja'Marr Chase", 'Jaxon Smith-Njigba', 'George Pickens'}

COLOR_UNDER = '#1b5e20'
COLOR_OVER  = '#b71c1c'
COLOR_FAIR  = '#1a237e'

@st.cache_data
def load_data():
    df = pd.read_parquet('predictions.parquet')
    df['apy_m']           = df['apy'] / 1_000_000
    df['predicted_apy_m'] = df['predicted_apy'] / 1_000_000
    df['verdict'] = df['delta_m'].apply(
        lambda x: 'Underpaid' if x > 2 else ('Overpaid' if x < -2 else 'Fair Value')
    )
    return df

df = load_data()

if 'is_legit_market' in df.columns:
    market_base = df[df['is_legit_market'] == 1].copy()
else:
    market_base = df[df['is_rookie_deal'] == 0].copy()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    "<p style='font-family:DM Serif Display,serif;font-size:1.3rem;"
    "color:#f7f6f3;margin-bottom:0.2rem;'>Filters</p>",
    unsafe_allow_html=True
)
st.sidebar.caption("Leaderboards always show legitimate starters on market deals only.")
st.sidebar.divider()

positions = st.sidebar.multiselect(
    "Position",
    options=['QB', 'RB', 'WR', 'TE'],
    default=['QB', 'RB', 'WR', 'TE']
)

if 'team' in df.columns:
    all_teams = ['All teams'] + sorted(df['team'].dropna().unique().tolist())
    selected_team = st.sidebar.selectbox("Team", options=all_teams)
else:
    selected_team = 'All teams'

age_min = int(df['age'].min())
age_max = int(df['age'].max())
age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))

if 'archetype' in df.columns:
    all_archetypes = sorted(df['archetype'].dropna().unique().tolist())
    selected_archetypes = st.sidebar.multiselect("Archetype", options=all_archetypes, default=[])
else:
    selected_archetypes = []

apy_min = float(df['apy_m'].min())
apy_max = float(df['apy_m'].max())
apy_range = st.sidebar.slider("Actual APY ($M)", apy_min, apy_max, (apy_min, apy_max), step=0.5)

st.sidebar.divider()
show_all = st.sidebar.checkbox("Include rookie / backup deals in Overview", value=False)

# ── Filters ────────────────────────────────────────────────────────────────────
def apply_filters(data):
    d = data.copy()
    if positions:
        d = d[d['position'].isin(positions)]
    if selected_team != 'All teams' and 'team' in d.columns:
        d = d[d['team'] == selected_team]
    d = d[(d['age'] >= age_range[0]) & (d['age'] <= age_range[1])]
    if selected_archetypes and 'archetype' in d.columns:
        d = d[d['archetype'].isin(selected_archetypes)]
    d = d[(d['apy_m'] >= apy_range[0]) & (d['apy_m'] <= apy_range[1])]
    return d

overview_df = apply_filters(df if show_all else market_base)
market_df   = apply_filters(market_base)

underpaid_df = market_df[market_df['delta_m'] >= UNDERPAID_MIN].copy()
overpaid_df  = market_df[
    (market_df['delta_m'] <= OVERPAID_MAX) &
    (~market_df['position'].isin(OVERPAID_EXCL)) &
    (~market_df['player_name'].isin(FUTURE_VALUE_EXCEPTIONS))
].copy()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="nfl-header">NFL Contract Analyzer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="nfl-sub">XGBoost + K-Means trained on 2021–2024 NFL stats and nflverse contracts. '
    'Covers QB, RB, WR, and TE. Predictions are stats-based — injuries, scheme fit, '
    'and negotiation leverage are not modeled. Active contracts signed 2020 and later only.</p>',
    unsafe_allow_html=True
)
st.markdown('<hr class="nfl-rule">', unsafe_allow_html=True)

# ── KPIs ───────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Market deal players", len(market_df))
k2.metric("Underpaid",  len(underpaid_df),
          help=f"Model predicts player worth at least ${UNDERPAID_MIN:.1f}M more than contract")
k3.metric("Overpaid", len(overpaid_df),
          help=f"Contract exceeds model by at least ${abs(OVERPAID_MAX):.1f}M (QB / WR / TE only)")
k4.metric("Fair value", len(market_df) - len(underpaid_df) - len(overpaid_df))
k5.metric("Avg contract gap", f"${market_df['delta_m'].mean():+.1f}M")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", "Leaderboards", "Player Lookup", "Position Analysis"
])

# Plotly base layout
PLOT_LAYOUT = dict(
    plot_bgcolor='#ffffff',
    paper_bgcolor='#ffffff',
    font=dict(family='DM Sans, sans-serif', size=12, color='#333'),
    xaxis=dict(gridcolor='#f0ede6', linecolor='#e0ddd6'),
    yaxis=dict(gridcolor='#f0ede6', linecolor='#e0ddd6'),
    margin=dict(t=40, b=20, l=10, r=10),
)

COLOR_MAP = {
    'Underpaid':  COLOR_UNDER,
    'Overpaid':   COLOR_OVER,
    'Fair Value': COLOR_FAIR,
}

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown('<p class="section-label">Actual vs Predicted APY</p>',
                    unsafe_allow_html=True)
        fig = px.scatter(
            overview_df,
            x='apy_m', y='predicted_apy_m',
            color='verdict',
            color_discrete_map=COLOR_MAP,
            hover_name='player_name',
            hover_data={
                'position':        True,
                'archetype':       True if 'archetype' in overview_df.columns else False,
                'age':             True,
                'apy_m':           ':.1f',
                'predicted_apy_m': ':.1f',
                'delta_m':         ':.1f',
                'guarantee_pct':   ':.0%' if 'guarantee_pct' in overview_df.columns else False,
                'verdict':         False,
            },
            labels={'apy_m': 'Actual APY ($M)', 'predicted_apy_m': 'Predicted APY ($M)'},
            height=480,
        )
        max_val = max(overview_df['apy_m'].max(), overview_df['predicted_apy_m'].max()) + 2
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode='lines',
            line=dict(dash='dot', color='#bbb', width=1.5),
            name='Fair value',
        ))
        fig.update_layout(
            **PLOT_LAYOUT,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0,
                        font=dict(size=11)),
            title=dict(
                text='Points above the diagonal are underpaid · Below are overpaid',
                font=dict(size=11, color='#999'),
                x=0, pad=dict(b=10)
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<p class="section-label">Contract Gap Distribution</p>',
                    unsafe_allow_html=True)
        fig2 = px.histogram(
            overview_df, x='delta_m', nbins=35,
            color='verdict',
            color_discrete_map=COLOR_MAP,
            labels={'delta_m': 'Predicted − Actual APY ($M)'},
            height=210,
        )
        fig2.add_vline(x=0, line_dash='dot', line_color='#999', line_width=1.5,
                       annotation_text='Fair value',
                       annotation_font=dict(size=10, color='#999'),
                       annotation_position='top right')
        fig2.update_layout(**PLOT_LAYOUT, showlegend=False, bargap=0.06)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<p class="section-label">Avg Gap by Position</p>',
                    unsafe_allow_html=True)
        pos_summary = (overview_df.groupby('position')
                       .agg(avg_delta=('delta_m', 'mean'), n=('player_name', 'count'))
                       .reset_index().sort_values('avg_delta', ascending=False))
        fig3 = px.bar(
            pos_summary, x='position', y='avg_delta',
            color='avg_delta',
            color_continuous_scale=[COLOR_OVER, '#e8e6e0', COLOR_UNDER],
            color_continuous_midpoint=0,
            text=pos_summary['avg_delta'].apply(lambda x: f'{x:+.1f}M'),
            labels={'avg_delta': 'Avg Delta ($M)', 'position': ''},
            height=200,
        )
        fig3.update_traces(textposition='outside',
                           textfont=dict(size=11, family='IBM Plex Mono'))
        fig3.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    if 'archetype' in overview_df.columns:
        st.markdown('<p class="section-label">Avg Contract Gap by Archetype</p>',
                    unsafe_allow_html=True)
        arch_pivot = (overview_df.groupby(['position', 'archetype'])['delta_m']
                      .mean().reset_index()
                      .pivot(index='archetype', columns='position', values='delta_m'))
        fig4 = px.imshow(
            arch_pivot,
            color_continuous_scale=[COLOR_OVER, '#f7f6f3', COLOR_UNDER],
            color_continuous_midpoint=0,
            text_auto='.1f',
            aspect='auto',
            height=320,
        )
        fig4.update_layout(
            **PLOT_LAYOUT,
            title=dict(
                text='Green = underpaid relative to archetype peers · Red = overpaid',
                font=dict(size=11, color='#999'), x=0
            )
        )
        st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LEADERBOARDS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.caption(
        f"{len(market_df)} legitimate starters on market deals shown. "
        "Rookie deals and backup-only seasons excluded. "
        "Underpaid = model predicts at least $0.5M above contract. "
        "Overpaid = contract exceeds model by at least $1M (QB / WR / TE only). "
        "RBs excluded from overpaid — insufficient training sample."
    )

    def make_leaderboard_df(data, ascending=True, n=15):
        if data.empty:
            return pd.DataFrame()
        lb = (data.nsmallest(n, 'delta_m') if ascending
              else data.nlargest(n, 'delta_m')).copy()
        lb['Actual APY']    = lb['apy_m'].apply(lambda x: f"${x:.1f}M")
        lb['Predicted APY'] = lb['predicted_apy_m'].apply(lambda x: f"${x:.1f}M")
        lb['Delta']         = lb['delta_m'].apply(
            lambda x: f"+${x:.1f}M" if x >= 0 else f"-${abs(x):.1f}M")
        lb['Guarantee %']   = lb['guarantee_pct'].apply(
            lambda x: f"{x*100:.0f}%") if 'guarantee_pct' in lb.columns else '—'
        lb['Age']   = lb['age'].apply(lambda x: f"{x:.0f}")
        lb['Games'] = lb['games'].apply(
            lambda x: f"{x:.0f}") if 'games' in lb.columns else '—'

        cols = ['player_name', 'position', 'Age', 'Games',
                'Actual APY', 'Predicted APY', 'Delta', 'Guarantee %']
        if 'archetype' in lb.columns:
            cols.insert(4, 'archetype')
        if 'team' in lb.columns:
            cols.insert(2, 'team')
        cols = [c for c in cols if c in lb.columns or c in [
            'player_name', 'position', 'Age', 'Games',
            'Actual APY', 'Predicted APY', 'Delta', 'Guarantee %']]

        return lb[cols].rename(columns={
            'player_name': 'Player', 'position': 'Pos',
            'archetype': 'Archetype', 'team': 'Team'})

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<p class="section-label">Most Underpaid</p>', unsafe_allow_html=True)
        st.dataframe(
            make_leaderboard_df(underpaid_df, ascending=False, n=15),
            use_container_width=True, hide_index=True, height=520)

    with col_r:
        st.markdown('<p class="section-label">Most Overpaid</p>', unsafe_allow_html=True)
        if overpaid_df.empty:
            st.info("No players meet the overpaid threshold with current filters.")
        else:
            st.dataframe(
                make_leaderboard_df(overpaid_df, ascending=True, n=15),
                use_container_width=True, hide_index=True, height=520)

    st.divider()
    st.markdown('<p class="section-label">By Position</p>', unsafe_allow_html=True)
    pos_tabs = st.tabs(['QB', 'RB', 'WR', 'TE'])
    for i, pos in enumerate(['QB', 'RB', 'WR', 'TE']):
        with pos_tabs[i]:
            pos_market = market_df[market_df['position'] == pos]
            if pos_market.empty:
                st.info(f"No {pos} players with current filters.")
                continue

            pos_under = pos_market[pos_market['delta_m'] >= UNDERPAID_MIN]
            pos_over  = pos_market[pos_market['delta_m'] <= OVERPAID_MAX] \
                        if pos not in OVERPAID_EXCL else pd.DataFrame()

            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<p class="section-label">Most Underpaid</p>',
                            unsafe_allow_html=True)
                if pos_under.empty:
                    st.info(f"No underpaid {pos} players with current filters.")
                else:
                    st.dataframe(
                        make_leaderboard_df(pos_under, ascending=False, n=10),
                        use_container_width=True, hide_index=True, height=380)
            with c2:
                st.markdown('<p class="section-label">Most Overpaid</p>',
                            unsafe_allow_html=True)
                if pos in OVERPAID_EXCL:
                    st.caption(
                        f"RB overpaid predictions excluded — only {len(pos_market)} "
                        "market-deal RBs in training data."
                    )
                elif pos_over.empty:
                    st.info(f"No {pos} players meet the overpaid threshold.")
                else:
                    st.dataframe(
                        make_leaderboard_df(pos_over, ascending=True, n=10),
                        use_container_width=True, hide_index=True, height=380)

            fig_pos = px.scatter(
                pos_market, x='apy_m', y='predicted_apy_m',
                color='verdict', color_discrete_map=COLOR_MAP,
                hover_name='player_name',
                hover_data={'apy_m': ':.1f', 'predicted_apy_m': ':.1f',
                            'delta_m': ':.1f', 'age': True, 'verdict': False},
                labels={'apy_m': 'Actual APY ($M)', 'predicted_apy_m': 'Predicted ($M)'},
                height=300,
            )
            max_v = max(pos_market['apy_m'].max(), pos_market['predicted_apy_m'].max()) + 2
            fig_pos.add_trace(go.Scatter(
                x=[0, max_v], y=[0, max_v], mode='lines',
                line=dict(dash='dot', color='#bbb', width=1),
                name='Fair value', showlegend=False))
            fig_pos.update_layout(**PLOT_LAYOUT, showlegend=False)
            st.plotly_chart(fig_pos, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PLAYER LOOKUP
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    col_search, col_compare = st.columns([2, 1])
    with col_search:
        all_players = sorted(df['player_name'].dropna().unique().tolist())
        selected_player = st.selectbox("Player", all_players)
    with col_compare:
        compare_player = st.selectbox(
            "Compare with (optional)",
            ['None'] + [p for p in all_players if p != selected_player])

    def render_player_card(player_name, data):
        rows = data[data['player_name'] == player_name]
        if rows.empty:
            st.warning(f"No data for {player_name}")
            return
        p     = rows.iloc[0]
        delta = p['delta_m']
        pos   = p['position']

        if delta > 2:
            card_cls, label, color = 'verdict-under', 'Underpaid', COLOR_UNDER
        elif delta < -2:
            card_cls, label, color = 'verdict-over', 'Overpaid', COLOR_OVER
        else:
            card_cls, label, color = 'verdict-fair', 'Fair Value', COLOR_FAIR

        gap_str = f'+${delta:.1f}M' if delta >= 0 else f'-${abs(delta):.1f}M'
        st.markdown(f"""
        <div class="verdict-card {card_cls}">
            <div class="verdict-label" style="color:{color};">{label}</div>
            <div class="verdict-sub">Gap of {gap_str} vs market prediction</div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Position",   p['position'])
        m2.metric("Age",        f"{p['age']:.0f}")
        m3.metric("Team",       p.get('team', 'N/A'))
        m4.metric("Actual APY", f"${p['apy_m']:.1f}M")
        m5.metric("Predicted",  f"${p['predicted_apy_m']:.1f}M")
        m6.metric("Guarantee",  f"{p.get('guarantee_pct', 0)*100:.0f}%")

        if 'archetype' in p and pd.notna(p.get('archetype')):
            st.caption(f"Archetype: {p['archetype']}")

        st.divider()
        s1, s2, s3 = st.columns(3)

        with s1:
            st.markdown('<p class="section-label">Season Context</p>',
                        unsafe_allow_html=True)
            st.metric("Games",        f"{p.get('games', 0):.0f}")
            st.metric("Seasons",      f"{p.get('seasons', 0):.0f}")
            st.metric("Health %",     f"{p.get('games_played_pct', 0)*100:.0f}%")
            st.metric("Avg Health %", f"{p.get('avg_games_played_pct', 0)*100:.0f}%")
            st.metric("Playoff Games",f"{p.get('playoff_games', 0):.0f}")
            st.metric("Playoff Rate", f"{p.get('playoff_rate', 0)*100:.0f}%")

        with s2:
            st.markdown('<p class="section-label">Recent Production (injury-adj)</p>',
                        unsafe_allow_html=True)
            if pos == 'QB':
                st.metric("Pass Yds",  f"{p.get('passing_yards_adj', 0):,.0f}")
                st.metric("Pass TDs",  f"{p.get('passing_tds_adj', 0):.0f}")
                st.metric("Comp %",    f"{p.get('completion_pct', 0)*100:.1f}%")
                st.metric("Y/Att",     f"{p.get('yards_per_attempt', 0):.1f}")
                st.metric("Rush Yds",  f"{p.get('rushing_yards_adj', 0):,.0f}")
                st.metric("Rush TDs",  f"{p.get('rushing_tds_adj', 0):.0f}")
            elif pos == 'RB':
                st.metric("Rush Yds",   f"{p.get('rushing_yards_adj', 0):,.0f}")
                st.metric("Rush TDs",   f"{p.get('rushing_tds_adj', 0):.0f}")
                st.metric("Y/Carry",    f"{p.get('yards_per_carry', 0):.1f}")
                st.metric("Rec Yds",    f"{p.get('receiving_yards_adj', 0):,.0f}")
                st.metric("Receptions", f"{p.get('receptions_adj', 0):.0f}")
                st.metric("Catch %",    f"{p.get('catch_rate', 0)*100:.1f}%")
            else:
                st.metric("Rec Yds",    f"{p.get('receiving_yards_adj', 0):,.0f}")
                st.metric("Rec TDs",    f"{p.get('receiving_tds_adj', 0):.0f}")
                st.metric("Receptions", f"{p.get('receptions_adj', 0):.0f}")
                st.metric("Targets",    f"{p.get('targets', 0):.0f}")
                st.metric("Y/Target",   f"{p.get('yards_per_target', 0):.1f}")
                st.metric("Catch %",    f"{p.get('catch_rate', 0)*100:.1f}%")

        with s3:
            st.markdown('<p class="section-label">Career Peaks</p>',
                        unsafe_allow_html=True)
            if pos == 'QB':
                st.metric("Peak Pass Yds",  f"{p.get('peak_passing_yards_adj', 0):,.0f}")
                st.metric("Peak Pass TDs",  f"{p.get('peak_passing_tds_adj', 0):.0f}")
                st.metric("Best Pass Yd/G", f"{p.get('best_passing_yards_pg', 0):.0f}")
            elif pos == 'RB':
                st.metric("Peak Rush Yds",  f"{p.get('peak_rushing_yards_adj', 0):,.0f}")
                st.metric("Peak Rush TDs",  f"{p.get('peak_rushing_tds_adj', 0):.0f}")
                st.metric("Best Rush Yd/G", f"{p.get('best_rushing_yards_pg', 0):.0f}")
            else:
                st.metric("Peak Rec Yds",   f"{p.get('peak_receiving_yards_adj', 0):,.0f}")
                st.metric("Peak Rec TDs",   f"{p.get('peak_receiving_tds_adj', 0):.0f}")
                st.metric("Best Rec Yd/G",  f"{p.get('best_receiving_yards_pg', 0):.0f}")

            st.markdown('<p class="section-label" style="margin-top:1rem;">Contract</p>',
                        unsafe_allow_html=True)
            st.metric("FA Year", f"{p.get('fa_year', 'N/A')}")
            draft = p.get('draft_round', 8)
            st.metric("Draft Round",
                      f"Round {draft:.0f}" if float(draft) < 8 else "Undrafted")

        if 'archetype' in p and pd.notna(p.get('archetype')):
            st.divider()
            st.markdown(f'<p class="section-label">Similar Players — {p["archetype"]}</p>',
                        unsafe_allow_html=True)
            peers = df[
                (df['position'] == pos) &
                (df['archetype'] == p['archetype']) &
                (df['player_name'] != player_name)
            ].nlargest(6, 'apy_m')[
                ['player_name', 'apy_m', 'predicted_apy_m', 'delta_m', 'age']
            ].copy()
            peers.columns = ['Player', 'Actual APY ($M)', 'Predicted ($M)', 'Delta ($M)', 'Age']
            peers['Actual APY ($M)'] = peers['Actual APY ($M)'].apply(lambda x: f"${x:.1f}M")
            peers['Predicted ($M)']  = peers['Predicted ($M)'].apply(lambda x: f"${x:.1f}M")
            peers['Delta ($M)']      = peers['Delta ($M)'].apply(
                lambda x: f"+${x:.1f}M" if x >= 0 else f"-${abs(x):.1f}M")
            peers['Age'] = peers['Age'].apply(lambda x: f"{x:.0f}")
            st.dataframe(peers, use_container_width=True, hide_index=True)

    if compare_player == 'None':
        render_player_card(selected_player, df)
    else:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown(f"**{selected_player}**")
            render_player_card(selected_player, df)
        with col_p2:
            st.markdown(f"**{compare_player}**")
            render_player_card(compare_player, df)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — POSITION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    pos_select = st.selectbox("Position", ['QB', 'RB', 'WR', 'TE'])
    pos_data   = apply_filters(df[df['position'] == pos_select])
    pos_market = (pos_data[pos_data['is_legit_market'] == 1]
                  if 'is_legit_market' in pos_data.columns
                  else pos_data[pos_data['is_rookie_deal'] == 0])

    if pos_select == 'RB':
        st.caption(
            "RB contract predictions are less reliable — only 16 market-deal RBs "
            "in training data. Use stats as context rather than relying on delta values."
        )

    col_x, col_y = st.columns(2)
    with col_x:
        st.markdown('<p class="section-label">APY vs Age</p>', unsafe_allow_html=True)
        fig_age = px.scatter(
            pos_market, x='age', y='apy_m',
            color='delta_m',
            color_continuous_scale=[COLOR_OVER, '#f0ede6', COLOR_UNDER],
            color_continuous_midpoint=0,
            hover_name='player_name',
            hover_data={'apy_m': ':.1f', 'predicted_apy_m': ':.1f',
                        'delta_m': ':.1f', 'age': True},
            labels={'age': 'Age', 'apy_m': 'APY ($M)', 'delta_m': 'Delta ($M)'},
            size='apy_m', size_max=18, height=360,
        )
        fig_age.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig_age, use_container_width=True)

    with col_y:
        if 'guarantee_pct' in pos_market.columns:
            st.markdown('<p class="section-label">Guarantee % vs APY</p>',
                        unsafe_allow_html=True)
            fig_guar = px.scatter(
                pos_market, x='guarantee_pct', y='apy_m',
                color='delta_m',
                color_continuous_scale=[COLOR_OVER, '#f0ede6', COLOR_UNDER],
                color_continuous_midpoint=0,
                hover_name='player_name',
                hover_data={'apy_m': ':.1f', 'delta_m': ':.1f',
                            'guarantee_pct': ':.0%'},
                labels={'guarantee_pct': 'Guarantee %', 'apy_m': 'APY ($M)'},
                height=360,
            )
            fig_guar.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig_guar, use_container_width=True)

    if 'archetype' in pos_market.columns:
        st.markdown('<p class="section-label">Archetype Breakdown</p>',
                    unsafe_allow_html=True)
        arch_data = (pos_market.groupby('archetype')
                     .agg(n=('player_name','count'), avg_apy=('apy_m','mean'),
                          avg_predicted=('predicted_apy_m','mean'),
                          avg_delta=('delta_m','mean'), avg_age=('age','mean'),
                          avg_guarantee=('guarantee_pct','mean'))
                     .reset_index().sort_values('avg_apy', ascending=False))
        arch_data['avg_apy']       = arch_data['avg_apy'].apply(lambda x: f"${x:.1f}M")
        arch_data['avg_predicted'] = arch_data['avg_predicted'].apply(lambda x: f"${x:.1f}M")
        arch_data['avg_delta']     = arch_data['avg_delta'].apply(
            lambda x: f"+${x:.1f}M" if x >= 0 else f"-${abs(x):.1f}M")
        arch_data['avg_age']       = arch_data['avg_age'].apply(lambda x: f"{x:.0f}")
        arch_data['avg_guarantee'] = arch_data['avg_guarantee'].apply(lambda x: f"{x*100:.0f}%")
        arch_data.columns = ['Archetype', 'N', 'Avg APY', 'Avg Predicted',
                             'Avg Delta', 'Avg Age', 'Avg Guarantee']
        st.dataframe(arch_data, use_container_width=True, hide_index=True)

    st.markdown('<p class="section-label">All Players — Sortable</p>',
                unsafe_allow_html=True)
    display_cols = ['player_name', 'age', 'apy_m', 'predicted_apy_m',
                    'delta_m', 'verdict', 'guarantee_pct', 'games', 'seasons']
    if 'archetype' in pos_data.columns: display_cols.append('archetype')
    if 'team' in pos_data.columns:     display_cols.append('team')

    display = pos_data[[c for c in display_cols if c in pos_data.columns]].copy()
    display = display.rename(columns={
        'player_name': 'Player', 'age': 'Age', 'apy_m': 'Actual APY',
        'predicted_apy_m': 'Predicted', 'delta_m': 'Delta', 'verdict': 'Verdict',
        'guarantee_pct': 'Guarantee %', 'games': 'Games', 'seasons': 'Seasons',
        'archetype': 'Archetype', 'team': 'Team'
    })
    for col, fmt in [
        ('Actual APY',  lambda x: f"${x:.1f}M"),
        ('Predicted',   lambda x: f"${x:.1f}M"),
        ('Delta',       lambda x: f"+${x:.1f}M" if x >= 0 else f"-${abs(x):.1f}M"),
        ('Guarantee %', lambda x: f"{x*100:.0f}%"),
        ('Age',         lambda x: f"{x:.0f}"),
    ]:
        if col in display.columns:
            display[col] = display[col].apply(fmt)
    st.dataframe(display, use_container_width=True, hide_index=True, height=450)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:#aaa;font-size:0.72rem;letter-spacing:0.05em;'>"
    "NFL CONTRACT ANALYZER &nbsp;·&nbsp; XGBoost + K-Means &nbsp;·&nbsp; "
    "nflverse + nfl_data_py &nbsp;·&nbsp; QB / RB / WR / TE &nbsp;·&nbsp; "
    "Not affiliated with any NFL organization"
    "</p>",
    unsafe_allow_html=True
)