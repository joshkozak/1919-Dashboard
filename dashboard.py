import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np
from typing import Optional, Dict
import json
import os

# Add these function definitions here
def check_data_quality(df: pd.DataFrame) -> None:
    missing_values = df.isnull().sum()
    if missing_values.any():
        st.warning("Missing values detected in the dataset")
        st.write(missing_values[missing_values > 0])

    duplicates = df.duplicated().sum()
    if duplicates > 0:
        st.warning(f"Found {duplicates} duplicate entries")

def filter_options(options: list, search_term: str) -> list:
    if not search_term:
        return options
    return [opt for opt in options if search_term.lower() in str(opt).lower()]

def calculate_trends(df: pd.DataFrame, metric: str) -> Dict:
    trends = {
        'growth_rate': 0,
        'seasonality': [],
        'forecast': []
    }
    if not df.empty:
        values = df[metric].values
        if len(values) > 1:
            trends['growth_rate'] = ((values[-1] - values[0]) / values[0]) * 100
            # Add simple moving average for trend
            trends['forecast'] = np.convolve(values, np.ones(3)/3, mode='valid').tolist()
    return trends

def save_preferences() -> None:
    with open("user_preferences.json", "w") as f:
        json.dump(st.session_state.user_preferences, f)


# Set page config for dark mode
st.set_page_config(page_title="27-Month Rolling Sales Dashboard", layout="wide")
st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: #ffffff;
        }
        .stApp {
            background-color: #0e1117;
        }
        .css-1d391kg, .css-ffhzg2 {
            background-color: #262730;
        }
        .st-bx, .st-dg, .st-cj, .st-bo, .st-bn {
            background-color: #1f2128;
        }
    </style>
""", unsafe_allow_html=True)

# Add configuration and session state management
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {
        'default_metric': 'Net Price',
        'chart_type': 'line',
        'show_summary': True
    }

# Add configuration sidebar
with st.sidebar.expander("⚙️ Dashboard Settings", expanded=False):
    st.session_state.user_preferences['default_metric'] = st.selectbox(
        "Default Metric",
        ["Net Price", "Case Equivs", "Units Sold"],
        index=0
    )
    st.session_state.user_preferences['chart_type'] = st.selectbox(
        "Chart Type",
        ["line", "bar", "area"],
        index=0
    )
    st.session_state.user_preferences['show_summary'] = st.checkbox(
        "Show Summary Statistics",
        value=True
    )

# Add data validation function
def validate_excel_structure(df: pd.DataFrame) -> tuple[bool, str]:
    required_columns = ["Year", "Month", "Item Names", "Distributors", "State", "Case Equivs", "Units Sold", "Net Price"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    return True, "Data structure is valid"

# Enhanced data loading with validation
@st.cache_data(ttl=3600)
def load_data() -> Optional[pd.DataFrame]:
    try:
        # Get the directory containing the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct path to Excel file
        file_path = os.path.join(script_dir, "27_Month_rolling.xlsx")
        
        with st.spinner("Loading and validating data..."):
            excel_file = pd.ExcelFile(file_path)
            df = excel_file.parse("Rolling Periods 27 Month")
            
            # Validate data structure
            is_valid, message = validate_excel_structure(df)
            if not is_valid:
                st.error(message)
                return None
            
            # Data cleaning and preprocessing
            df["Year"] = pd.to_datetime(df["Year"]).dt.year
            df["FullDate"] = pd.to_datetime(df["Year"].astype(str) + '-' + df["Month"].astype(str) + '-01')
            
            # Add derived metrics
            df["Revenue per Case"] = df["Net Price"] / df["Case Equivs"]
            df["Revenue per Unit"] = df["Net Price"] / df["Units Sold"]
            
            return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load data
df = load_data()
if df is None:
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")

def multiselect_with_select_all(label, options, key):
    select_all = st.sidebar.checkbox(f"Select All {label}", value=True, key=f"{key}_all")
    if select_all:
        return options
    else:
        # Add search box for filters
        search_term = st.sidebar.text_input(f"Search {label}", key=f"search_{key}")
        filtered_options = filter_options(options, search_term)
        return st.sidebar.multiselect(label=f"Select {label}:", options=filtered_options, key=key)

all_items = sorted(df["Item Names"].unique())
all_distributors = sorted(df["Distributors"].unique())
all_states = sorted(df["State"].unique())

selected_items = multiselect_with_select_all("Item(s)", all_items, "items")
selected_distributors = multiselect_with_select_all("Distributor(s)", all_distributors, "distributors")
selected_states = multiselect_with_select_all("State(s)", all_states, "states")

# Date filter
min_year = int(df["Year"].min())
max_year = int(df["Year"].max())
date_range = st.sidebar.slider("Select Year Range:", min_value=min_year, max_value=max_year, value=(min_year, max_year), step=1)

# Filtered data
filtered_df = df[
    (df["Item Names"].isin(selected_items)) &
    (df["Distributors"].isin(selected_distributors)) &
    (df["State"].isin(selected_states)) &
    (df["Year"] >= date_range[0]) &
    (df["Year"] <= date_range[1])
]

# Check data quality
check_data_quality(filtered_df)

# Create monthly summary
monthly_summary = (
    filtered_df.groupby([filtered_df["Year"], filtered_df["Month"]])
    .agg({"Case Equivs": "sum", "Units Sold": "sum", "Net Price": "sum"})
    .reset_index()
)
monthly_summary["Date"] = pd.to_datetime(
    monthly_summary["Year"].astype(str) + '-' + 
    monthly_summary["Month"].astype(str) + '-01'
)

# Create tabs
tab1, tab2 = st.tabs(["Dashboard", "Data View"])

# Dashboard Tab
with tab1:
    st.title("27-Month Rolling Sales Dashboard")
    
    # KPIs
    total_cases = filtered_df["Case Equivs"].sum()
    total_units = filtered_df["Units Sold"].sum()
    total_revenue = filtered_df["Net Price"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Case Equivs", f"{total_cases:,.2f}")
    col2.metric("Total Units Sold", f"{int(total_units):,}")
    col3.metric("Total Revenue", f"${total_revenue:,.2f}")

    # Add trend analysis
    if st.checkbox("Show Trend Analysis"):
        trends = calculate_trends(monthly_summary, st.session_state.user_preferences['default_metric'])
        st.metric("Growth Rate", f"{trends['growth_rate']:.2f}%")
        
        if trends['forecast']:
            forecast_fig = px.line(
                y=trends['forecast'],
                title="Trend Forecast",
                labels={'value': 'Forecast', 'index': 'Time Period'}
            )
            forecast_fig.update_layout(paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', font_color='white')
            st.plotly_chart(forecast_fig)

    # Time Series Plot
    fig = px.line(monthly_summary, x="Date", y=["Case Equivs", "Units Sold", "Net Price"],
                  labels={"value": "Metric", "Date": "Date"},
                  title="Monthly Sales Metrics Over Time")
    fig.update_layout(
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        font_color='white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all", label="All")
            ])
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # Add correlation analysis
    if st.checkbox("Show Correlation Analysis"):
        correlation = monthly_summary[["Case Equivs", "Units Sold", "Net Price"]].corr()
        st.write("Correlation Matrix")
        st.dataframe(correlation.style.background_gradient())

# Data View Tab
with tab2:
    st.title("Data Analysis View")
    
    if st.session_state.user_preferences['show_summary']:
        st.subheader("Summary Statistics")
        summary_stats = filtered_df.describe()
        st.dataframe(summary_stats)
    
    st.subheader("Filtered Data")
    st.dataframe(filtered_df)
    
    # Add download button for filtered data
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name=f'sales_data_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv',
    )

# Add footer with additional information
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.markdown("### About this Dashboard")
    st.markdown("""
    - Data is refreshed hourly
    - Last update: {}
    - Contact support: support@example.com
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

with col2:
    st.markdown("### Quick Actions")
    if st.button("Save Preferences"):
        save_preferences()
        st.success("Preferences saved!")
    if st.button("Reset Preferences"):
        st.session_state.user_preferences = {
            'default_metric': 'Net Price',
            'chart_type': 'line',
            'show_summary': True
        }
        st.success("Preferences reset!")

# Add data refresh mechanism
if st.button("Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()







