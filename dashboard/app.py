import sys
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import joblib
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import src.helpers as helpers

# Page Configuration
st.set_page_config(
    page_title='Intelligent Road Safety ML | UML501',
    page_icon='🚦',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Apply Custom Theme CSS
css_path = ROOT / 'dashboard/style.css'
if css_path.exists():
    st.markdown(f'<style>\n{css_path.read_text(encoding="utf-8")}\n</style>', unsafe_allow_html=True)

pio.templates.default = 'plotly_white'
CUSTOM_PALETTE = ['#0d9488', '#f59e0b', '#ef4444', '#3b82f6', '#8b5cf6', '#06b6d4']
px.defaults.color_discrete_sequence = CUSTOM_PALETTE

# Human-Readable Label Mappings for STATS19 Codes
WEATHER_MAP = {
    1: 'Fine no high winds (1)',
    2: 'Raining no high winds (2)',
    3: 'Snowing no high winds (3)',
    4: 'Fine + high winds (4)',
    5: 'Raining + high winds (5)',
    6: 'Snowing + high winds (6)',
    7: 'Fog or mist (7)',
    8: 'Other (8)',
    9: 'Unknown (9)',
    -1: 'Missing / Out of range (-1)'
}

LIGHT_MAP = {
    1: 'Daylight (1)',
    4: 'Darkness - lights lit (4)',
    5: 'Darkness - lights unlit (5)',
    6: 'Darkness - no lighting (6)',
    7: 'Darkness - lighting unknown (7)',
    -1: 'Missing / Out of range (-1)'
}

ROAD_TYPE_MAP = {
    1: 'Roundabout (1)',
    2: 'One way street (2)',
    3: 'Dual carriageway (3)',
    6: 'Single carriageway (6)',
    7: 'Slip road (7)',
    9: 'Unknown (9)',
    12: 'One way street/Slip road (12)'
}

SURFACE_MAP = {
    1: 'Dry (1)',
    2: 'Wet or damp (2)',
    3: 'Snow (3)',
    4: 'Frost or ice (4)',
    5: 'Flood over 3cm deep (5)',
    6: 'Oil or diesel (6)',
    7: 'Mud (7)',
    9: 'Unknown (9)',
    -1: 'Missing / Out of range (-1)'
}

AREA_MAP = {
    1: 'Urban (1)',
    2: 'Rural (2)',
    3: 'Unallocated (3)',
    -1: 'Missing / Unknown (-1)'
}

VEHICLE_TYPE_MAP = {
    1: 'Pedal cycle (1)',
    2: 'Motorcycle <=50cc (2)',
    3: 'Motorcycle <=125cc (3)',
    4: 'Motorcycle 125cc-500cc (4)',
    5: 'Motorcycle >500cc (5)',
    8: 'Taxi / Private hire (8)',
    9: 'Car (9)',
    10: 'Minibus (8-16 pass) (10)',
    11: 'Bus or coach (>=17 pass) (11)',
    16: 'Horse / Ridden animal (16)',
    17: 'Agricultural vehicle (17)',
    18: 'Tram (18)',
    19: 'Van / Goods <=3.5t (19)',
    20: 'Goods 3.5t-7.5t (20)',
    21: 'Goods >=7.5t (21)',
    22: 'Mobility scooter (22)',
    23: 'Electric motorcycle (23)',
    33: 'Motor Caravan (33)',
    90: 'Other vehicle (90)',
    97: 'Motorcycle unknown cc (97)',
    98: 'Goods vehicle unknown (98)',
    99: 'Unknown vehicle type (99)',
    -1: 'Missing / Out of range (-1)'
}

DAY_MAP = {
    0: 'Monday (0)',
    1: 'Tuesday (1)',
    2: 'Wednesday (2)',
    3: 'Thursday (3)',
    4: 'Friday (4)',
    5: 'Saturday (5)',
    6: 'Sunday (6)'
}

MONTH_MAP = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}

# Cached Data & Model Loaders
@st.cache_data
def read_csv(path):
    return pd.read_csv(path)

@st.cache_resource
def get_model():
    return helpers.load_classifier()

@st.cache_resource
def get_regressor():
    return helpers.load_regressor()

@st.cache_data
def get_meta():
    return helpers.load_metadata()

def has(path):
    return path.exists()

# Sidebar Branding & Navigation
st.sidebar.markdown('''
<div class="sidebar-brand-card">
    <span class="course-tag">UML501 · Machine Learning</span>
    <h2>ROAD SAFETY ML</h2>
    <p>Predictive severity modeling & spatial hotspot analysis with UK STATS19</p>
</div>
''', unsafe_allow_html=True)

nav_options = [
    'Executive Overview',
    'Data Explorer',
    'Regional Risk Map',
    'Risk Predictor',
    'Model Comparison',
    'Regression & Trends',
    'Pattern Mining',
    'Academic Dossier'
]

# Map friendly labels to original page keys for backward compatibility
page_selection = st.sidebar.radio(
    'Navigation Menu',
    nav_options,
    index=0
)

st.sidebar.markdown('''
<div class="sidebar-status-box">
    <div style="margin-bottom: 4px;">
        <span class="status-dot"></span>
        <strong style="color: #ffffff;">System Status: Active</strong>
    </div>
    <div style="color: #94a3b8; font-size: 0.72rem;">
        Dataset: STATS19 (2021–2023)<br>
        Models: Scikit-Learn Pipeline Loaded<br>
        Framework: Streamlit & Plotly Engine
    </div>
</div>
''', unsafe_allow_html=True)

data_path = ROOT / 'data/processed/cleaned_accidents.csv'

# ==============================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==============================================================================
if page_selection == 'Executive Overview':
    st.markdown('''
    <div class="page-header-card">
        <span class="page-header-badge">Project Dashboard · College Subject Presentation</span>
        <h1>Intelligent Road Safety Decision Support</h1>
        <p>A comprehensive machine learning investigation into road collision severity patterns, environmental risk drivers, and geographic accident clusters utilizing UK Department for Transport STATS19 data (2021–2023).</p>
    </div>
    ''', unsafe_allow_html=True)

    if has(data_path):
        df = read_csv(data_path)
        meta = get_meta() if has(ROOT / 'models/metadata.json') else {}
        hot_path = ROOT / 'outputs/tables/hotspots.csv'
        hot = read_csv(hot_path) if has(hot_path) else pd.DataFrame()

        # Top Metric Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Accidents Analysed', f'{len(df):,}', help='Total cleaned collisions in analysis sample')
        fatal_pct = (df.severity == 'Fatal').mean() * 100
        m2.metric('Fatal Share', f'{fatal_pct:.2f}%', delta='Rare Severity Class', delta_color='inverse')
        serious_pct = (df.severity == 'Serious').mean() * 100
        m3.metric('Serious Share', f'{serious_pct:.1f}%', delta='High Impact Class', delta_color='inverse')
        m4.metric('Regional Clusters', len(hot), help='K-Means regional collision partitions; not density hotspots')

        # Best Model Callout Card
        if meta and 'metrics' in meta:
            b_metrics = meta['metrics']
            st.markdown(f'''
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 5px solid #0d9488; border-radius: 10px; padding: 1rem 1.4rem; margin: 1.2rem 0; box-shadow: 0 2px 8px rgba(15,23,42,0.05);">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <span style="font-size: 0.76rem; font-weight: 700; text-transform: uppercase; color: #0d9488; letter-spacing: 0.05em;">Selected Classifier (Evaluated on Untouched 30K Final Test Set)</span>
                        <h3 style="margin: 0; color: #0f172a; font-size: 1.25rem;">{meta.get("best_model", "Classifier")}</h3>
                    </div>
                    <div style="display: flex; gap: 1.5rem; flex-wrap: wrap;">
                        <div>
                            <span style="font-size: 0.74rem; color: #64748b; display: block;">Overall Accuracy</span>
                            <strong style="font-size: 1.05rem; color: #0f172a;">{float(b_metrics.get("accuracy", 0)):.1%}</strong>
                        </div>
                        <div>
                            <span style="font-size: 0.74rem; color: #64748b; display: block;">Fatal Recall</span>
                            <strong style="font-size: 1.05rem; color: #0d9488;">{float(b_metrics.get("fatal_recall", 0)):.1%}</strong>
                        </div>
                        <div>
                            <span style="font-size: 0.74rem; color: #64748b; display: block;">Serious Recall</span>
                            <strong style="font-size: 1.05rem; color: #f59e0b;">{float(b_metrics.get("serious_recall", 0)):.1%}</strong>
                        </div>
                        <div>
                            <span style="font-size: 0.74rem; color: #64748b; display: block;">Macro F1 Score</span>
                            <strong style="font-size: 1.05rem; color: #3b82f6;">{float(b_metrics.get("macro_f1", 0)):.3f}</strong>
                        </div>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

        # 4-Stage Project Architecture Pipeline Banner
        st.markdown('''
        <div style="margin: 1.5rem 0 1rem 0;">
            <h3 style="margin: 0 0 0.6rem 0; font-size: 1.1rem; color: #0f172a;">Project End-to-End Pipeline</h3>
            <div class="workflow-stepper">
                <div class="workflow-step active">
                    <div class="step-number">Stage 01</div>
                    <div class="step-title">Ingest</div>
                    <div class="step-desc">STATS19 collision, casualty, and vehicle relational joins.</div>
                </div>
                <div class="workflow-step active">
                    <div class="step-number">Stage 02</div>
                    <div class="step-title">Prepare</div>
                    <div class="step-desc">Bounding filters, median/mode imputation, feature engineering.</div>
                </div>
                <div class="workflow-step active">
                    <div class="step-number">Stage 03</div>
                    <div class="step-title">Model</div>
                    <div class="step-desc">Balanced classification, K-Means clustering, forecasting &amp; rules.</div>
                </div>
                <div class="workflow-step active">
                    <div class="step-number">Stage 04</div>
                    <div class="step-title">Deploy</div>
                    <div class="step-desc">8-view Streamlit dashboard &amp; hypothetical scenario simulator.</div>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # Overview Visuals Row
        c1, c2 = st.columns([1.2, 1])
        with c1:
            fig_pie = px.pie(
                df,
                names='severity',
                title='Class Distribution (Severe Imbalance)',
                color='severity',
                color_discrete_map={'Slight': '#0d9488', 'Serious': '#f59e0b', 'Fatal': '#ef4444'},
                hole=0.45
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#ffffff', width=2)))
            fig_pie.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=350)
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.markdown('''
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1.4rem; height: 350px; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <h4 style="margin: 0 0 0.5rem 0; color: #0f172a;">Why Class Imbalance Matters</h4>
                    <p style="font-size: 0.88rem; color: #475569; line-height: 1.5; margin-bottom: 0.8rem;">
                        In road safety, <strong>Fatal collisions comprise only 1.48%</strong> of all incidents. A naive baseline model that blindly predicts "Slight" for every single crash achieves <strong>76.7% accuracy</strong>, yet misses 100% of fatalities.
                    </p>
                    <p style="font-size: 0.88rem; color: #475569; line-height: 1.5; margin: 0;">
                        Our methodology uses cost-sensitive learning to emphasize fatal-event sensitivity. These retrospective scenario estimates are for academic decision support, not real-time early warning.
                    </p>
                </div>
            </div>
            ''', unsafe_allow_html=True)
    else:
        st.info('Cleaned dataset not found. Please run notebooks 01 and 02 to generate data artifacts.')

# ==============================================================================
# PAGE 2: INTERACTIVE DATA EXPLORER
# ==============================================================================
elif page_selection == 'Data Explorer':
    st.markdown('''
    <div class="page-header-card">
        <span class="page-header-badge">Exploratory Data Analysis</span>
        <h1>Interactive Collision Data Explorer</h1>
        <p>Slice, filter, and inspect STATS19 records across temporal, environmental, and geographic dimensions to uncover empirical collision trends.</p>
    </div>
    ''', unsafe_allow_html=True)

    if not has(data_path):
        st.info('Processed data not found. Please run notebook 01 first.')
    else:
        df = read_csv(data_path)

        with st.expander('🔍 Dynamic Filters & Slice Controls', expanded=True):
            f1, f2, f3, f4, f5 = st.columns(5)
            years = f1.multiselect('Collision Year', sorted(df.collision_year.unique()), default=sorted(df.collision_year.unique()))
            severities = f2.multiselect('Severity', df.severity.unique(), default=list(df.severity.unique()))
            weathers = f3.multiselect('Weather', sorted(df.weather_label.dropna().unique()), default=sorted(df.weather_label.dropna().unique()))
            roads = f4.multiselect('Road Type', sorted(df.road_type_label.dropna().unique()), default=sorted(df.road_type_label.dropna().unique()))
            areas = f5.multiselect('Area Type', sorted(df.area_label.dropna().unique()), default=sorted(df.area_label.dropna().unique()))

        filtered_df = df[
            df.collision_year.isin(years) &
            df.severity.isin(severities) &
            df.weather_label.isin(weathers) &
            df.road_type_label.isin(roads) &
            df.area_label.isin(areas)
        ]

        # Filter KPI Summary
        k1, k2, k3, k4 = st.columns(4)
        k1.metric('Filtered Collisions', f'{len(filtered_df):,}')
        pct_active = (len(filtered_df) / len(df)) * 100 if len(df) else 0
        k2.metric('% of Dataset', f'{pct_active:.1f}%')
        k3.metric('Fatal Incidents', f'{(filtered_df.severity == "Fatal").sum():,}')
        k4.metric('Serious Incidents', f'{(filtered_df.severity == "Serious").sum():,}')

        tab_temporal, tab_environmental, tab_records = st.tabs([
            '🕒 Temporal Distributions',
            '🌧️ Road & Environmental Factors',
            '📋 Raw Sample Records'
        ])

        with tab_temporal:
            t1, t2 = st.columns(2)
            with t1:
                fig_hour = px.histogram(
                    filtered_df, x='hour', color='severity',
                    title='Accidents by Hour of Day (Rush Hour Spikes)',
                    color_discrete_map={'Slight': '#0d9488', 'Serious': '#f59e0b', 'Fatal': '#ef4444'},
                    barmode='stack', nbins=24
                )
                fig_hour.update_layout(xaxis_title='Hour (0-23)', yaxis_title='Collisions', height=360)
                st.plotly_chart(fig_hour, use_container_width=True)

            with t2:
                fig_day = px.histogram(
                    filtered_df, x='day_name', color='severity',
                    title='Accidents by Day of Week',
                    color_discrete_map={'Slight': '#0d9488', 'Serious': '#f59e0b', 'Fatal': '#ef4444'},
                    category_orders={'day_name': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}
                )
                fig_day.update_layout(xaxis_title='Day of Week', yaxis_title='Collisions', height=360)
                st.plotly_chart(fig_day, use_container_width=True)

            fig_month = px.histogram(
                filtered_df, x='month', color='severity',
                title='Accidents by Month of Year',
                color_discrete_map={'Slight': '#0d9488', 'Serious': '#f59e0b', 'Fatal': '#ef4444'},
                nbins=12
            )
            fig_month.update_layout(xaxis_title='Month (1-12)', yaxis_title='Collisions', height=320)
            st.plotly_chart(fig_month, use_container_width=True)

        with tab_environmental:
            e1, e2 = st.columns(2)
            with e1:
                fig_weather = px.histogram(
                    filtered_df, x='weather_label', color='severity',
                    title='Accidents by Weather Condition',
                    color_discrete_map={'Slight': '#0d9488', 'Serious': '#f59e0b', 'Fatal': '#ef4444'}
                )
                fig_weather.update_layout(xaxis_title='Weather Condition', yaxis_title='Collisions', height=360)
                st.plotly_chart(fig_weather, use_container_width=True)

            with e2:
                fig_speed = px.histogram(
                    filtered_df, x='speed_limit', color='severity',
                    title='Accidents by Posted Speed Limit (mph)',
                    color_discrete_map={'Slight': '#0d9488', 'Serious': '#f59e0b', 'Fatal': '#ef4444'}
                )
                fig_speed.update_layout(xaxis_title='Speed Limit (mph)', yaxis_title='Collisions', height=360)
                st.plotly_chart(fig_speed, use_container_width=True)

        with tab_records:
            st.dataframe(
                filtered_df[['collision_index', 'collision_year', 'date', 'time', 'severity', 'weather_label', 'road_type_label', 'speed_limit', 'vehicle_count', 'casualty_count']].head(100),
                use_container_width=True
            )
            st.caption('Displaying top 100 rows matching active filter criteria.')

# ==============================================================================
# PAGE 3: SPATIAL HOTSPOT MAP
# ==============================================================================
elif page_selection == 'Regional Risk Map':
    st.markdown('''
    <div class="page-header-card">
        <span class="page-header-badge">Unsupervised Spatial Clustering</span>
        <h1>Regional Collision Clustering & Composite Ranking</h1>
        <p>K-Means regional clustering of projected collision coordinates, with a descriptive composite ranking overlaid on an interactive OpenStreetMap layer.</p>
    </div>
    ''', unsafe_allow_html=True)

    path = ROOT / 'outputs/tables/hotspots.csv'
    if not path.exists():
        st.info('Hotspots data not found. Please run notebook 03 first.')
    else:
        hot = read_csv(path)
        max_n = max(1, min(10, len(hot)))

        h_col1, h_col2 = st.columns([1, 2])
        with h_col1:
            n = st.slider('Select Top-N Regional Clusters to Display', 1, max_n, min(5, max_n))
            top = hot.iloc[0]
            st.markdown(f'''
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1.2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <span style="font-size: 0.74rem; font-weight: 700; text-transform: uppercase; color: #ef4444; letter-spacing: 0.05em;">Highest Composite Score</span>
                <h3 style="margin: 0.2rem 0; color: #0f172a;">Region {int(top.cluster)}</h3>
                <div style="margin-top: 0.8rem; font-size: 0.88rem; color: #334155;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                        <span>Total Accidents:</span> <strong>{int(top.accidents):,}</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                        <span>Serious/Fatal Share:</span> <strong style="color: #ef4444;">{top.severe_share:.1%}</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                        <span>Avg Casualties:</span> <strong>{top.avg_casualties:.2f}</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>Composite Score:</span> <strong style="color: #0d9488;">{top.risk_score:.1f} / 100</strong>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

            st.markdown('''
            <div style="margin-top: 1rem; font-size: 0.78rem; color: #64748b; line-height: 1.45;">
                <strong>Regional Composite Formula:</strong><br>
                <em>Risk Score</em> combines normalized crash volume, severe injury proportion, and casualty density with equal weighting:
                <br><code>Score = (1/3)*(Accidents_norm + SevereShare_norm + AvgCas_norm) * 100</code>
                <br><span style="font-size: 0.74rem;">Normalized via Min-Max scaling: (x - Min) / (Max - Min)</span>
            </div>
            ''', unsafe_allow_html=True)

        with h_col2:
            st.dataframe(
                hot.head(n)[['cluster', 'accidents', 'severe_share', 'avg_casualties', 'risk_score', 'latitude', 'longitude']],
                use_container_width=True
            )

        # Map Rendering
        st.markdown('### Interactive Geographic Regional Layer')
        mp = folium.Map([hot.latitude.mean(), hot.longitude.mean()], zoom_start=6, tiles='OpenStreetMap')

        markers = folium.FeatureGroup(name='Regional Cluster Centroids')
        for idx, r in hot.head(n).iterrows():
            popup_html = f'''
            <div style="font-family: sans-serif; font-size: 12px; width: 180px;">
                <strong style="color: #0f172a; font-size: 13px;">Region #{int(r.cluster)}</strong><br>
                <hr style="margin: 4px 0;">
                <b>Total Collisions:</b> {int(r.accidents):,}<br>
                <b>Severe Share:</b> {r.severe_share:.1%}<br>
                <b>Avg Casualties:</b> {r.avg_casualties:.2f}<br>
                <b>Risk Score:</b> <span style="color: #0d9488; font-weight: bold;">{r.risk_score:.1f}</span>
            </div>
            '''
            folium.CircleMarker(
                [r.latitude, r.longitude],
                popup=folium.Popup(popup_html, max_width=200),
                radius=8 + (r.risk_score / 12),
                color='#ef4444' if idx == 0 else '#0d9488',
                fill=True,
                fill_color='#ef4444' if idx == 0 else '#0d9488',
                fill_opacity=0.75,
                weight=2
            ).add_to(markers)
        markers.add_to(mp)

        if has(data_path):
            loc = read_csv(data_path)[['latitude', 'longitude']].dropna().sample(min(20000, len(read_csv(data_path))), random_state=42)
            HeatMap(loc.values.tolist(), name='Accident Density Heatmap', radius=8, blur=10, min_opacity=0.3).add_to(mp)

        folium.LayerControl().add_to(mp)
        st_folium(mp, use_container_width=True, height=520)

# ==============================================================================
# PAGE 4: REAL-TIME RISK SIMULATOR
# ==============================================================================
elif page_selection == 'Risk Predictor':
    st.markdown('''
    <div class="page-header-card">
        <span class="page-header-badge">Machine Learning Inference Engine</span>
        <h1>Hypothetical Collision Severity Simulator</h1>
        <p>Enter assumed collision, road, and weather conditions to estimate severity probabilities from historical STATS19 records.</p>
    </div>
    ''', unsafe_allow_html=True)

    if not has(ROOT / 'models/best_classifier.joblib') or not has(ROOT / 'models/metadata.json'):
        st.info('Model artifact not found. Please run notebook 02 to train and export the best model.')
        st.stop()

    meta = get_meta()
    options = meta['dropdown_options']

    if 'scenario' not in st.session_state:
        st.session_state.scenario = 'Custom'

    st.markdown('##### ⚡ Quick Scenario Presets')
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    if s_col1.button('🌧️ Rainy Night, Rural Road', use_container_width=True):
        st.session_state.scenario = 'Rain'
    if s_col2.button('☀️ Sunny Weekday, Urban City', use_container_width=True):
        st.session_state.scenario = 'City'
    if s_col3.button('🛣️ Wet Dual Carriageway, Rush Hour', use_container_width=True):
        st.session_state.scenario = 'Motorway'
    if s_col4.button('⚙️ Custom Configuration', use_container_width=True):
        st.session_state.scenario = 'Custom'

    scenario = st.session_state.scenario
    default_hour = {'Rain': 22, 'City': 8, 'Motorway': 17}.get(scenario, 12)
    default_weather = {'Rain': 2, 'City': 1, 'Motorway': 2}.get(scenario)
    default_light = {'Rain': 6, 'City': 1, 'Motorway': 1}.get(scenario)
    default_road = {'Rain': 6, 'City': 6, 'Motorway': 3}.get(scenario)
    default_surface = {'Rain': 2, 'City': 1, 'Motorway': 2}.get(scenario)
    default_area = {'Rain': 2, 'City': 1, 'Motorway': 2}.get(scenario)
    default_speed = {'Rain': 60, 'City': 30, 'Motorway': 70}.get(scenario)
    default_day = {'Rain': 4, 'City': 1, 'Motorway': 4}.get(scenario, 2)

    with st.form('predict_form'):
        st.markdown('### Parameter Inputs')
        p1, p2, p3 = st.columns(3)
        hour = p1.slider('Hour of Day (24-hr)', 0, 23, default_hour)
        day = p2.selectbox('Day of Week', range(7), index=default_day, format_func=lambda x: DAY_MAP.get(x, str(x)))
        month = p3.selectbox('Month', range(1, 13), index=5, format_func=lambda x: MONTH_MAP.get(x, str(x)))

        is_weekend = day >= 5
        is_night = hour >= 22 or hour < 6
        rush_hour = hour in [7, 8, 9, 16, 17, 18]

        c_a, c_b = st.columns(2)
        with c_a:
            weather_choices = options.get('weather_conditions', [1])
            w_idx = weather_choices.index(default_weather) if default_weather in weather_choices else 0
            weather_conditions = st.selectbox('Weather Conditions', weather_choices, index=w_idx, format_func=lambda x: WEATHER_MAP.get(x, f'Code {x}'))

            light_choices = options.get('light_conditions', [1])
            l_idx = light_choices.index(default_light) if default_light in light_choices else 0
            light_conditions = st.selectbox('Lighting Conditions', light_choices, index=l_idx, format_func=lambda x: LIGHT_MAP.get(x, f'Code {x}'))

            road_choices = options.get('road_type', [6])
            r_idx = road_choices.index(default_road) if default_road in road_choices else 0
            road_type = st.selectbox('Road Type', road_choices, index=r_idx, format_func=lambda x: ROAD_TYPE_MAP.get(x, f'Code {x}'))

        with c_b:
            surface_choices = options.get('road_surface_conditions', [1])
            s_idx = surface_choices.index(default_surface) if default_surface in surface_choices else 0
            road_surface_conditions = st.selectbox('Road Surface Conditions', surface_choices, index=s_idx, format_func=lambda x: SURFACE_MAP.get(x, f'Code {x}'))

            area_choices = options.get('urban_or_rural_area', [1])
            a_idx = area_choices.index(default_area) if default_area in area_choices else 0
            urban_or_rural_area = st.selectbox('Urban or Rural Area', area_choices, index=a_idx, format_func=lambda x: AREA_MAP.get(x, f'Code {x}'))

            vehicle_choices = options.get('first_vehicle_type', [9])
            v_idx = vehicle_choices.index(9) if 9 in vehicle_choices else 0
            first_vehicle_type = st.selectbox('Primary Vehicle Type', vehicle_choices, index=v_idx, format_func=lambda x: VEHICLE_TYPE_MAP.get(x, f'Type {x}'))

        c_s1, c_s2 = st.columns(2)
        speed_choices = options.get('speed_limit', [30])
        sp_idx = speed_choices.index(default_speed) if default_speed in speed_choices else 0
        speed_limit = c_s1.selectbox('Posted Speed Limit (mph)', speed_choices, index=sp_idx)
        vehicle_count = c_s2.number_input('Total Number of Vehicles Involved', 1, 20, 2)

        submit = st.form_submit_button('Run Scenario Severity Estimate', use_container_width=True)

    if submit:
        row = {
            'hour': hour,
            'day_of_week': day,
            'month': month,
            'is_weekend': int(is_weekend),
            'is_night': int(is_night),
            'rush_hour': int(rush_hour),
            'speed_limit': speed_limit,
            'vehicle_count': vehicle_count,
            'weather_conditions': weather_conditions,
            'light_conditions': light_conditions,
            'road_type': road_type,
            'road_surface_conditions': road_surface_conditions,
            'urban_or_rural_area': urban_or_rural_area,
            'first_vehicle_type': first_vehicle_type
        }

        model = get_model()
        input_df = pd.DataFrame([row])[meta['features']]
        prediction = model.predict(input_df)[0]

        # Badge Styling by Severity
        badge_class = 'risk-badge-slight'
        badge_text = '🟢 Slight Injury Estimated'
        if prediction == 'Serious':
            badge_class = 'risk-badge-serious'
            badge_text = '🟠 Serious Injury Estimated'
        elif prediction == 'Fatal':
            badge_class = 'risk-badge-fatal'
            badge_text = '🔴 Fatal Outcome Estimated'

        st.markdown(f'''
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.5rem; margin: 1.5rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="font-size: 0.8rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Prediction Result</span>
                    <h2 style="margin: 0; color: #0f172a; font-size: 1.8rem;">{prediction} Severity</h2>
                </div>
                <div class="risk-badge {badge_class}">
                    {badge_text}
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(input_df)[0]
            classes = meta['classes']

            r_col1, r_col2 = st.columns([1.2, 1])
            with r_col1:
                prob_df = pd.DataFrame({'Severity': classes, 'Probability': probs})
                fig_bar = px.bar(
                    prob_df,
                    x='Severity',
                    y='Probability',
                    text=prob_df['Probability'].apply(lambda x: f'{x:.1%}'),
                    title='Class Probability Breakdown',
                    color='Severity',
                    color_discrete_map={'Slight': '#0d9488', 'Serious': '#f59e0b', 'Fatal': '#ef4444'}
                )
                fig_bar.update_layout(yaxis=dict(range=[0, 1]), height=320)
                st.plotly_chart(fig_bar, use_container_width=True)

            with r_col2:
                # Weighted risk score
                risk = float(sum(prob * weight for prob, weight in zip(probs, [95, 60, 15])))
                gauge_color = '#0d9488'
                if risk > 45:
                    gauge_color = '#f59e0b'
                if risk > 70:
                    gauge_color = '#ef4444'

                fig_gauge = go.Figure(go.Indicator(
                    mode='gauge+number',
                    value=risk,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': 'Scenario Severity Risk Score (0-100)', 'font': {'size': 16, 'color': '#0f172a'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#94a3b8'},
                        'bar': {'color': gauge_color},
                        'bgcolor': '#f8fafc',
                        'borderwidth': 1,
                        'bordercolor': '#e2e8f0',
                        'steps': [
                            {'range': [0, 45], 'color': 'rgba(13, 148, 136, 0.15)'},
                            {'range': [45, 70], 'color': 'rgba(245, 158, 11, 0.15)'},
                            {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                        ]
                    }
                ))
                fig_gauge.update_layout(height=320, margin=dict(t=40, b=20, l=30, r=30))
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.caption('Scenario Severity Risk Score is a scenario-specific weighted severity index: 95 × P(Fatal) + 60 × P(Serious) + 15 × P(Slight), distinct from the geographic macro-zone Composite Spatial Risk Index on Page 3. Class probabilities come from the selected historical-data model and are not calibrated real-world risk estimates.')

            # Feature Influences
            tip_path = ROOT / 'outputs/tables/selected_model_feature_effects.csv'
            if tip_path.exists():
                tips = read_csv(tip_path).head(4)
                def clean_feat_name(f):
                    name = f.replace('num__', '').replace('cat__', '')
                    for prefix, mapper in [
                        ('first_vehicle_type_', VEHICLE_TYPE_MAP),
                        ('light_conditions_', LIGHT_MAP),
                        ('weather_conditions_', WEATHER_MAP),
                        ('road_surface_conditions_', SURFACE_MAP),
                        ('road_type_', ROAD_TYPE_MAP),
                        ('urban_or_rural_area_', AREA_MAP),
                    ]:
                        if name.startswith(prefix):
                            code_str = name[len(prefix):]
                            try:
                                code_val = int(code_str)
                                label = mapper.get(code_val, f'Code {code_val}')
                                col_title = prefix.rstrip('_').replace('_', ' ').title()
                                return f'{col_title}: {label}'
                            except ValueError:
                                pass
                    return name.replace('_', ' ').title()
                feat_tags = ' '.join([f'<span style="background: #f1f5f9; border: 1px solid #cbd5e1; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; margin-right: 5px; color: #1e293b;">{clean_feat_name(f)}</span>' for f in tips.feature.tolist()])
                st.markdown(f'''
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.9rem; margin-top: 1rem;">
                    <strong style="color: #0f172a; font-size: 0.85rem;">Selected-model global effects:</strong> {feat_tags}
                </div>
                ''', unsafe_allow_html=True)

        st.caption('Academic Decision-Support Prototype. Estimates are statistical inferences based on historical STATS19 data and should not substitute for professional roadside safety audits.')

# ==============================================================================
# PAGE 5: MODEL COMPARISON & EVALUATION
# ==============================================================================
elif page_selection == 'Model Comparison':
    st.markdown('''
    <div class="page-header-card">
        <span class="page-header-badge">Benchmark & Validation</span>
        <h1>Comparative Classifier Evaluation</h1>
        <p>Eight classifiers compared on a validation split using class-specific recall, macro-averaged F1, and a safety-weighted selection criterion. The published model is then evaluated once on a final test split.</p>
    </div>
    ''', unsafe_allow_html=True)

    p_metrics = ROOT / 'outputs/tables/classification_metrics.csv'
    if p_metrics.exists():
        metrics_df = read_csv(p_metrics)

        tab_bench, tab_imbalance, tab_diagnostic, tab_sens = st.tabs([
            '🏆 Multi-Classifier Benchmark',
            '⚖️ Class Imbalance Mitigation',
            '📉 Diagnostic Curves & Errors',
            '🎯 Sensitivity & Specificity'
        ])

        with tab_bench:
            st.dataframe(metrics_df, use_container_width=True)

            fig_comp = px.bar(
                metrics_df,
                x='model',
                y=['fatal_recall', 'serious_recall', 'macro_f1', 'accuracy'],
                barmode='group',
                title='Multi-Metric Performance by Algorithm',
                color_discrete_sequence=['#ef4444', '#f59e0b', '#0d9488', '#64748b']
            )
            fig_comp.update_layout(xaxis_title='Model', yaxis_title='Score (0.0 to 1.0)', height=400)
            st.plotly_chart(fig_comp, use_container_width=True)

            st.markdown('''
            <div style="background: #f0fdf4; border-left: 4px solid #16a34a; padding: 0.85rem 1.1rem; border-radius: 6px; font-size: 0.88rem; color: #166534;">
                <strong>Validation Benchmark Protocol:</strong> Candidate classifiers are evaluated on the 24,000-instance validation split. Selection Score = 0.5 * Fatal Recall + 0.5 * Serious Recall + 0.5 * Macro F1. The selected Logistic Regression model is refitted on all 120,000 training records, achieving 65.0% Fatal Recall and 51.8% Accuracy on the untouched 30,000-record final test set (displayed on Page 1).
            </div>
            ''', unsafe_allow_html=True)

        with tab_imbalance:
            imb_path = ROOT / 'outputs/tables/imbalance_comparison.csv'
            if imb_path.exists():
                st.dataframe(read_csv(imb_path), use_container_width=True)
                st.markdown('''
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.1rem; margin-top: 1rem;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #0f172a;">Imbalance Mitigation Analysis</h4>
                    <ul style="font-size: 0.88rem; color: #334155; margin: 0; padding-left: 1.2rem; line-height: 1.6;">
                        <li><strong>Standard Logistic Regression:</strong> Validation comparison without class weighting.</li>
                        <li><strong>Class-Weighted Logistic Regression:</strong> Validation comparison using inverse-frequency class weights.</li>
                        <li><strong>SMOTE (Synthetic Minority Over-sampling):</strong> Applied only to a candidate-training subset and evaluated on validation data; it is not part of the final test-set evaluation.</li>
                    </ul>
                </div>
                ''', unsafe_allow_html=True)

        with tab_diagnostic:
            d1, d2 = st.columns(2)
            with d1:
                if (ROOT / 'outputs/figures/confusion_matrix.png').exists():
                    st.image(str(ROOT / 'outputs/figures/confusion_matrix.png'), caption='Figure: Confusion Matrix of Best Classifier')
            with d2:
                if (ROOT / 'outputs/figures/roc_curves.png').exists():
                    st.image(str(ROOT / 'outputs/figures/roc_curves.png'), caption='Figure: Multi-Class ROC Curves')

            d3, d4 = st.columns(2)
            with d3:
                if (ROOT / 'outputs/figures/feature_importance.png').exists():
                    st.image(str(ROOT / 'outputs/figures/feature_importance.png'), caption='Figure: Random Forest Gini Feature Importance')
            with d4:
                if (ROOT / 'outputs/figures/ann_loss_curve.png').exists():
                    st.image(str(ROOT / 'outputs/figures/ann_loss_curve.png'), caption='Figure: Artificial Neural Network (MLPClassifier) Training Loss')

            if (ROOT / 'outputs/figures/decision_tree.png').exists():
                st.image(str(ROOT / 'outputs/figures/decision_tree.png'), caption='Figure: Decision Tree Structure (Depth=3)')

        with tab_sens:
            sp_path = ROOT / 'outputs/tables/best_model_sensitivity_specificity.csv'
            if sp_path.exists():
                st.dataframe(read_csv(sp_path), use_container_width=True)
                st.caption('Sensitivity (True Positive Rate) vs Specificity (True Negative Rate) per severity tier.')
    else:
        st.info('Classifier metrics not found. Run notebook 02 first.')

# ==============================================================================
# PAGE 6: REGRESSION & TEMPORAL TRENDS
# ==============================================================================
elif page_selection == 'Regression & Trends':
    st.markdown('''
    <div class="page-header-card">
        <span class="page-header-badge">Continuous Modeling</span>
        <h1>Daily Collision Volume Forecasting</h1>
        <p>Estimating daily aggregate collision totals from full raw data, with a chronological holdout rather than a random future-leaking split.</p>
    </div>
    ''', unsafe_allow_html=True)

    reg_path = ROOT / 'outputs/tables/regression_metrics.csv'
    if reg_path.exists():
        reg_df = read_csv(reg_path)
        best_r = reg_df.sort_values('RMSE').iloc[0]

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric('Best Model', best_r.model)
        rc2.metric('R² Score', f'{best_r.R2:.3f}')
        rc3.metric('RMSE', f'{best_r.RMSE:.1f} crashes')
        rc4.metric('MAE', f'{best_r.MAE:.1f} crashes')

        st.dataframe(reg_df, use_container_width=True)

        g1, g2 = st.columns(2)
        with g1:
            if (ROOT / 'outputs/figures/regression_actual_predicted.png').exists():
                st.image(str(ROOT / 'outputs/figures/regression_actual_predicted.png'), caption='Figure: Actual vs. Predicted Daily Collisions')
        with g2:
            if (ROOT / 'outputs/figures/daily_trend.png').exists():
                st.image(str(ROOT / 'outputs/figures/daily_trend.png'), caption='Figure: Daily Collision Timeline & Rolling Trends')

        if (ROOT / 'outputs/figures/regression_residuals.png').exists():
            st.image(str(ROOT / 'outputs/figures/regression_residuals.png'), caption='Figure: Residuals of Polynomial Regression Model')

        # Interactive Daily Forecasting using best_regressor.joblib
        reg_model = get_regressor()
        if reg_model is not None:
            st.markdown('### 🔮 Interactive Daily Collision Forecast')
            st.caption(f'Simulate expected daily collision volume using the saved {best_r.model} model. Accuracy statistics refer to the chronological holdout period.')
            with st.form('reg_forecast_form'):
                rf1, rf2, rf3 = st.columns(3)
                f_day = rf1.selectbox('Day of Week', range(7), index=4, format_func=lambda x: DAY_MAP.get(x, str(x)))
                f_month = rf2.selectbox('Month of Year', range(1, 13), index=10, format_func=lambda x: MONTH_MAP.get(x, str(x)))
                f_weekend = int(f_day >= 5)
                rf3.info('Weekend: Yes' if f_weekend else 'Weekend: No (Weekday)')

                rf4, rf5 = st.columns(2)
                f_rain = rf4.slider('Expected Rain Share of Collisions (%)', 0, 100, 15) / 100.0
                f_night = rf5.slider('Expected Night Share of Collisions (%)', 0, 100, 20) / 100.0

                f_submit = st.form_submit_button('Run Daily Volume Forecast', use_container_width=True)
                if f_submit:
                    f_row = pd.DataFrame([{
                        'day': f_day,
                        'month': f_month,
                        'weekend': f_weekend,
                        'rain_share': f_rain,
                        'night_share': f_night
                    }])
                    f_pred = reg_model.predict(f_row)[0]
                    st.success(f'Estimated Daily Collisions: **{max(0.0, f_pred):.1f}** crashes (Baseline MAE: ±{best_r.MAE:.1f} crashes)')
    else:
        st.info('Regression metrics not found. Please run notebook 04 first.')

# ==============================================================================
# PAGE 7: ASSOCIATION RULE MINING
# ==============================================================================
elif page_selection == 'Pattern Mining':
    st.markdown('''
    <div class="page-header-card">
        <span class="page-header-badge">Unsupervised Pattern Discovery</span>
        <h1>Environmental Association Rule Mining</h1>
        <p>Uncovering descriptive environmental co-occurrence rules whose consequents are a single serious or fatal severity class.</p>
    </div>
    ''', unsafe_allow_html=True)

    rules_path = ROOT / 'outputs/tables/association_rules.csv'
    if rules_path.exists():
        rules_df = read_csv(rules_path)

        st.markdown('''
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.2rem;">
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem;">
                <strong style="color: #0d9488;">Support</strong>
                <p style="margin: 0.3rem 0 0 0; font-size: 0.8rem; color: #64748b;">The proportion of total collisions containing the combined itemset {A, B}.</p>
            </div>
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem;">
                <strong style="color: #f59e0b;">Confidence</strong>
                <p style="margin: 0.3rem 0 0 0; font-size: 0.8rem; color: #64748b;">Conditional probability: P(Severity | Conditions). How often B occurs when A is present.</p>
            </div>
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem;">
                <strong style="color: #ef4444;">Lift</strong>
                <p style="margin: 0.3rem 0 0 0; font-size: 0.8rem; color: #64748b;">Strength of association over independent chance. Lift &gt; 1 indicates positive co-occurrence; it is not a significance test or causal estimate.</p>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # Format rules for clean presentation
        def format_rule_set(s):
            import re
            items = re.findall(r"'([^']+)'", str(s))
            label_dict = {
                'light=1': 'Daylight', 'light=4': 'Dark (lights lit)', 'light=5': 'Dark (lights unlit)', 'light=6': 'Dark (no lights)', 'light=7': 'Dark (lighting unk)',
                'area=1': 'Urban', 'area=2': 'Rural',
                'road_type=1': 'Roundabout', 'road_type=2': 'One way', 'road_type=3': 'Dual carriageway', 'road_type=6': 'Single carriageway', 'road_type=7': 'Slip road',
                'weather=1': 'Fine weather', 'weather=2': 'Raining', 'weather=3': 'Snowing', 'weather=4': 'Fine+winds', 'weather=5': 'Rain+winds',
                'speed_band=low': 'Speed <=30mph', 'speed_band=medium': 'Speed 40-50mph', 'speed_band=high': 'Speed 60-70mph', 'speed_band=very_high': 'Speed >70mph',
                'severity=Fatal': 'Fatal Severity', 'severity=Serious': 'Serious Severity', 'severity=Slight': 'Slight Severity'
            }
            return ', '.join([label_dict.get(item, item) for item in items])

        display_rules = rules_df.copy()
        if 'antecedents' in display_rules.columns and 'consequents' in display_rules.columns:
            display_rules['antecedents'] = display_rules['antecedents'].apply(format_rule_set)
            display_rules['consequents'] = display_rules['consequents'].apply(format_rule_set)

        st.dataframe(display_rules, use_container_width=True)
    else:
        st.info('Association rules data not found. Please run notebook 05 first.')

# ==============================================================================
# PAGE 8: ACADEMIC PROJECT DOSSIER
# ==============================================================================
else:
    st.markdown('''
    <div class="page-header-card">
        <span class="page-header-badge">Academic Metadata & Governance</span>
        <h1>Project Dossier & Course Mapping</h1>
        <p>Institutional metadata, technical specifications, and syllabus course learning outcome (CLO) mappings for university examination and evaluation.</p>
    </div>
    ''', unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    with d1:
        st.markdown('''
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1.4rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <h3 style="margin-top: 0; color: #0f172a;">Course & Candidate Credentials</h3>
            <table style="width: 100%; font-size: 0.9rem; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #f1f5f9; height: 36px;">
                    <td style="color: #64748b;">Subject / Course:</td>
                    <td><strong>UML501: Machine Learning</strong></td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9; height: 36px;">
                    <td style="color: #64748b;">Program & Semester:</td>
                    <td><strong>B.Tech Computer Science & Engg (5th Sem)</strong></td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9; height: 36px;">
                    <td style="color: #64748b;">Student Candidate:</td>
                    <td><strong>Arnav Gupta<br>Jasmandeep Sharma</strong></td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9; height: 36px;">
                    <td style="color: #64748b;">University Roll Number:</td>
                    <td><strong>1024170057<br>1024170269</strong></td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9; height: 36px;">
                    <td style="color: #64748b;">Project Supervisor / Guide:</td>
                    <td><strong>Mr. Harjot Singh (Lab Instructor)</strong></td>
                </tr>
                <tr style="height: 36px;">
                    <td style="color: #64748b;">Institution:</td>
                    <td><strong>Thapar Institute of Engineering &amp; Technology</strong></td>
                </tr>
                <tr style="height: 36px;">
                    <td style="color: #64748b;">Academic Session:</td>
                    <td><strong>2026–2027</strong></td>
                </tr>
            </table>
        </div>
        ''', unsafe_allow_html=True)

    with d2:
        st.markdown('''
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1.4rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <h3 style="margin-top: 0; color: #0f172a;">Course Learning Outcomes (CLO)</h3>
            <div style="margin-bottom: 0.8rem;">
                <strong style="color: #0d9488;">CLO 1: Data Preprocessing & EDA</strong>
                <p style="margin: 0.2rem 0; font-size: 0.82rem; color: #475569;">Handling missing values, outlier bounding, temporal feature engineering, and exploratory data profiling.</p>
            </div>
            <div style="margin-bottom: 0.8rem;">
                <strong style="color: #3b82f6;">CLO 2: Supervised Learning & Imbalance Mitigation</strong>
                <p style="margin: 0.2rem 0; font-size: 0.82rem; color: #475569;">Comparative benchmarking of 8 classifiers, cost-sensitive weighting, SMOTE, and regression volume forecasting.</p>
            </div>
            <div>
                <strong style="color: #f59e0b;">CLO 3: Unsupervised Clustering & Neural Networks</strong>
                <p style="margin: 0.2rem 0; font-size: 0.82rem; color: #475569;">Spatial K-Means clustering, silhouette optimization, Apriori association rules, and MLPClassifier neural network training.</p>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown('''
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1.4rem; margin-top: 1.4rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
        <h3 style="margin-top: 0; color: #0f172a;">Technology Stack & Provenance</h3>
        <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 1rem;">
            <span style="background: #e0f2fe; color: #0369a1; padding: 4px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600;">Python 3.12</span>
            <span style="background: #e0f2fe; color: #0369a1; padding: 4px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600;">Scikit-Learn</span>
            <span style="background: #e0f2fe; color: #0369a1; padding: 4px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600;">Imbalanced-learn</span>
            <span style="background: #e0f2fe; color: #0369a1; padding: 4px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600;">MLxtend</span>
            <span style="background: #e0f2fe; color: #0369a1; padding: 4px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600;">Streamlit</span>
            <span style="background: #e0f2fe; color: #0369a1; padding: 4px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600;">Plotly Express</span>
            <span style="background: #e0f2fe; color: #0369a1; padding: 4px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600;">Matplotlib &amp; Seaborn</span>
            <span style="background: #e0f2fe; color: #0369a1; padding: 4px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600;">Folium &amp; GIS Leaflet</span>
            <span style="background: #e0f2fe; color: #0369a1; padding: 4px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600;">Pandas &amp; NumPy</span>
            <span style="background: #e0f2fe; color: #0369a1; padding: 4px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600;">Joblib</span>
        </div>
        <p style="font-size: 0.85rem; color: #64748b; line-height: 1.5; margin: 0;">
            Data Source: Open road casualty records published by the Department for Transport (DfT), UK, licensed under the Open Government Licence v3.0.
        </p>
    </div>
    ''', unsafe_allow_html=True)

