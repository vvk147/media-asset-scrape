import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
from company_data_comparison import analyze_company_data, get_api_usage_stats, set_api_keys
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv
import logging
import time

# Configure logging with a more concise format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Streamlit page config
st.set_page_config(
    page_title="Company Data Analysis",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables only once at startup
if "env_loaded" not in st.session_state:
    if os.getenv("STREAMLIT_DEPLOYMENT") == "true":
        # In Streamlit Cloud, use secrets
        logger.info("Running in Streamlit Cloud, using secrets")
        st.session_state["env_loaded"] = True
    else:
        # In local development, use .env file
        load_dotenv()
        logger.info("Running locally, loaded environment from .env file")
        st.session_state["env_loaded"] = True

def initialize_session_state():
    """Initialize session state variables"""
    if "SCRAPINGANT_API_KEY" not in st.session_state:
        st.session_state["SCRAPINGANT_API_KEY"] = os.getenv("SCRAPINGANT_API_KEY")
    if "EXA_API_KEY" not in st.session_state:
        st.session_state["EXA_API_KEY"] = os.getenv("EXA_API_KEY")

# Initialize session state
initialize_session_state()

# Add sidebar navigation and API key management
st.sidebar.title("Settings")

# API Key Management Section
with st.sidebar.expander("API Key Management", expanded=True):
    # Show current source of API keys
    if os.getenv("STREAMLIT_DEPLOYMENT") != "true":
        if os.getenv("SCRAPINGANT_API_KEY") and os.getenv("EXA_API_KEY"):
            st.success("✅ Using API keys from .env file")
            st.info("You can override these keys below if needed")
    
    new_scrapingant_key = st.text_input(
        "ScrapingAnt API Key",
        value=st.session_state.get("SCRAPINGANT_API_KEY", ""),
        type="password",
        help="Enter your ScrapingAnt API key"
    )
    
    new_exa_key = st.text_input(
        "Exa.ai API Key",
        value=st.session_state.get("EXA_API_KEY", ""),
        type="password",
        help="Enter your Exa.ai API key"
    )
    
    if st.button("Save API Keys"):
        st.session_state["SCRAPINGANT_API_KEY"] = new_scrapingant_key
        st.session_state["EXA_API_KEY"] = new_exa_key
        set_api_keys(new_scrapingant_key, new_exa_key)
        st.success("API keys saved successfully!")

st.sidebar.markdown("---")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Company Analysis", "API Analytics"])

def create_gauge_chart(value, title):
    """Create a gauge chart using plotly"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "lightgray"},
                {'range': [33, 66], 'color': "gray"},
                {'range': [66, 100], 'color': "darkgray"}
            ],
        }
    ))
    fig.update_layout(height=200)
    return fig

def display_company_data(company_data, show_header=True):
    """Display company data in a consistent format"""
    if show_header:
        st.header("Company Analysis Results")
    
    # Company Overview
    st.subheader("Company Overview")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if company_data["logo_url"]:
            try:
                st.markdown(
                    f"<img src='{company_data['logo_url']}' style='max-width: 200px; width: 100%;'>",
                    unsafe_allow_html=True
                )
            except Exception as e:
                logger.warning(f"Failed to load logo: {str(e)}")
                st.warning("⚠️ Company logo could not be loaded")
    
    with col2:
        st.write("**Company Name:**", company_data["name"])
        st.write("**Domain:**", company_data["domain"])
        if company_data["description"]:
            st.write("**Description:**", company_data["description"])
    
    # Product Lines
    if company_data.get("product_lines"):
        st.subheader("Product Lines")
        for product in company_data["product_lines"]:
            with st.expander(product["name"]):
                if product["description"]:
                    st.write(product["description"])
                else:
                    st.write("No description available")
    
    # Social Media Presence
    if company_data["social_links"]:
        st.subheader("Social Media Presence")
        for link in company_data["social_links"]:
            st.write(f"- [{link}]({link})")
    
    # Media Assets
    if company_data["media_assets"]:
        st.subheader("Media Assets")
        cols = st.columns(3)
        for i, asset in enumerate(company_data["media_assets"]):
            if asset["type"] == "image":
                with cols[i % 3]:
                    try:
                        caption = asset["alt"] if asset["alt"] else "No caption"
                        st.markdown(
                            f"<figure><img src='{asset['url']}' style='width: 100%'><figcaption>{caption}</figcaption></figure>",
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        logger.warning(f"Failed to load media asset: {str(e)}")
                        st.warning("⚠️ Media asset could not be loaded")

def display_analysis_history():
    """Display historical data with detailed view"""
    if not os.path.exists("company_data_history.csv"):
        st.info("No analysis history available yet.")
        return
        
    try:
        # Read CSV with proper date parsing using the recommended approach
        df = pd.read_csv("company_data_history.csv")
        # Convert scrape_date column to datetime after reading
        df['scrape_date'] = pd.to_datetime(df['scrape_date'], format='%Y-%m-%d %H:%M:%S')
        
        # Drop rows with invalid dates
        df = df.dropna(subset=['scrape_date'])
        
        if df.empty:
            st.warning("No valid data found in history.")
            return
        
        # Sort by date
        df = df.sort_values('scrape_date', ascending=False)
        
        # Add filters
        col1, col2 = st.columns(2)
        with col1:
            try:
                min_date = df['scrape_date'].min().date()
                max_date = df['scrape_date'].max().date()
                date_filter = st.date_input(
                    "Filter by date",
                    value=[min_date, max_date],
                    min_value=min_date,
                    max_value=max_date
                )
            except Exception as e:
                logger.error(f"Error setting date filter: {str(e)}")
                date_filter = [min_date, max_date]
        
        with col2:
            domain_filter = st.multiselect(
                "Filter by domain",
                options=sorted(df['domain'].dropna().unique())
            )
        
        # Apply filters
        mask = (df['scrape_date'].dt.date >= date_filter[0]) & (df['scrape_date'].dt.date <= date_filter[1])
        if domain_filter:
            mask = mask & (df['domain'].isin(domain_filter))
        
        filtered_df = df[mask]
        
        if filtered_df.empty:
            st.warning("No data found for the selected filters.")
            return
        
        # Display data with selection
        selection = st.data_editor(
            filtered_df[['scrape_date', 'name', 'domain', 'description']],
            hide_index=True,
            column_config={
                "scrape_date": st.column_config.DatetimeColumn(
                    "Analysis Date",
                    format="DD/MM/YYYY HH:mm"
                ),
                "name": st.column_config.TextColumn(
                    "Company Name",
                    width="medium"
                ),
                "domain": st.column_config.TextColumn(
                    "Domain",
                    width="medium"
                ),
                "description": st.column_config.TextColumn(
                    "Description",
                    width="large"
                )
            },
            use_container_width=True,
            num_rows="dynamic"
        )
        
        # If a row is selected
        if selection is not None and not selection.empty:
            try:
                # Get the full data for the selected row
                selected_domain = selection.iloc[0]['domain']
                selected_date = selection.iloc[0]['scrape_date']
                
                # Find the complete row in the original dataframe
                full_row = filtered_df[
                    (filtered_df['domain'] == selected_domain) & 
                    (filtered_df['scrape_date'] == selected_date)
                ].iloc[0]
                
                # Create company data dictionary with safe evaluation
                company_data = {
                    "name": str(full_row["name"]) if pd.notna(full_row["name"]) else "",
                    "domain": str(full_row["domain"]) if pd.notna(full_row["domain"]) else "",
                    "description": str(full_row["description"]) if pd.notna(full_row["description"]) else "",
                    "logo_url": str(full_row["logo_url"]) if pd.notna(full_row["logo_url"]) else "",
                    "social_links": eval(full_row["social_links"]) if pd.notna(full_row["social_links"]) else [],
                    "media_assets": eval(full_row["media_assets"]) if pd.notna(full_row["media_assets"]) else [],
                    "product_lines": eval(full_row["product_lines"]) if pd.notna(full_row["product_lines"]) else []
                }
                
                # Display the selected company data
                st.markdown("---")
                st.subheader("Selected Company Details")
                display_company_data(company_data, show_header=False)
                
            except Exception as e:
                logger.error(f"Error displaying selected company data: {str(e)}")
                st.error("Failed to display selected company data. The data might be in an incorrect format.")
    
    except Exception as e:
        logger.error(f"Error reading history file: {str(e)}")
        st.error(
            "Error reading history file. The file might be corrupted. "
            "You may need to delete it and start fresh."
        )

def display_api_analytics():
    """Display API usage analytics"""
    st.title("📊 API Usage Analytics")
    
    # Get API usage statistics
    usage_stats = get_api_usage_stats()
    
    if usage_stats["success"]:
        stats = usage_stats["stats"]
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Requests", f"{stats['total_requests']:,}")
        with col2:
            st.metric("Total Credits Used", f"{stats['total_credits_used']:,}")
        with col3:
            st.metric("Total Cost", f"${stats['total_cost']:.2f}")
        with col4:
            st.metric("Success Rate", f"{stats['success_rate']:.1f}%")
        
        st.markdown("---")
        
        # Usage over time
        st.subheader("Usage Over Time")
        
        # Convert dictionary to DataFrame for plotting with proper date handling
        usage_df = pd.DataFrame([
            {"date": pd.to_datetime(date), "requests": count}
            for date, count in stats['usage_by_day'].items()
        ])
        
        # Create line chart
        fig = px.line(
            usage_df, 
            x='date', 
            y='requests',
            title='Daily API Requests',
            labels={'date': 'Date', 'requests': 'Number of Requests'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cost over time with proper date handling
        cost_df = pd.DataFrame([
            {"date": pd.to_datetime(date), "cost": cost}
            for date, cost in stats['cost_by_day'].items()
        ])
        
        fig = px.line(
            cost_df, 
            x='date', 
            y='cost',
            title='Daily API Costs',
            labels={'date': 'Date', 'cost': 'Cost ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Usage breakdown
        st.subheader("Usage Breakdown")
        if os.path.exists("api_usage.csv"):
            # Read CSV with proper date handling
            df = pd.read_csv("api_usage.csv")
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter controls
            col1, col2 = st.columns(2)
            with col1:
                date_range = st.date_input(
                    "Date Range",
                    value=[
                        df['timestamp'].min().date(),
                        df['timestamp'].max().date()
                    ]
                )
            with col2:
                api_filter = st.multiselect(
                    "Filter by API",
                    options=df['api_name'].unique(),
                    default=df['api_name'].unique()
                )
            
            # Apply filters
            mask = (
                (df['timestamp'].dt.date >= date_range[0]) & 
                (df['timestamp'].dt.date <= date_range[1]) &
                (df['api_name'].isin(api_filter))
            )
            filtered_df = df[mask]
            
            # Display detailed table
            st.dataframe(
                filtered_df.style.format({
                    'timestamp': lambda x: pd.to_datetime(x).strftime('%Y-%m-%d %H:%M:%S'),
                    'cost': '${:.4f}',
                    'response_time': '{:.2f} s'
                }),
                hide_index=True
            )
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download API usage data as CSV",
                data=csv,
                file_name=f"api_usage_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    else:
        st.error(f"Failed to load API usage statistics: {usage_stats['error']}")

def main():
    try:
        # Load API keys from session state
        scrapingant_key = st.session_state.get("SCRAPINGANT_API_KEY")
        exa_key = st.session_state.get("EXA_API_KEY")
        
        # Store current page in session state to track changes
        if "current_page" not in st.session_state:
            st.session_state.current_page = page
        
        # Check if page has changed
        if st.session_state.current_page != page:
            # Clear all non-persistent session state variables
            for key in list(st.session_state.keys()):
                if key not in ["env_loaded", "SCRAPINGANT_API_KEY", "EXA_API_KEY", "current_page"]:
                    del st.session_state[key]
            st.session_state.current_page = page
        
        # Handle navigation based on sidebar selection
        if page == "Company Analysis":
            st.title("🏢 Company Data Analysis Tool")
            
            # Create tabs for Company Analysis
            tab1, tab2 = st.tabs(["New Analysis", "Analysis History"])
            
            with tab1:
                # Input section
                st.subheader("Enter Company URL")
                company_url = st.text_input(
                    "Company Website URL",
                    placeholder="https://example.com"
                )
                
                if st.button("Analyze Company", type="primary"):
                    if not scrapingant_key or not exa_key:
                        st.error("⚠️ Please set your API keys in the sidebar before proceeding.")
                        return
                        
                    if company_url:
                        with st.spinner("Analyzing company data..."):
                            try:
                                # Set API keys for analysis
                                set_api_keys(scrapingant_key, exa_key)
                                
                                results = analyze_company_data(company_url)
                                
                                if results["success"]:
                                    # Store results in session state
                                    st.session_state.analysis_results = results
                                    display_company_data(results["company_data"])
                                else:
                                    st.error(f"Analysis failed: {results['error']}")
                            except Exception as e:
                                st.error(f"An error occurred: {str(e)}")
                                logger.error(f"Analysis error: {str(e)}", exc_info=True)
                    else:
                        st.warning("Please enter a company URL")
            
            with tab2:
                display_analysis_history()
        
        else:  # API Analytics page
            # Create a container for API Analytics
            api_analytics_container = st.container()
            
            with api_analytics_container:
                display_api_analytics()
            
            # Add auto-refresh controls in sidebar
            st.sidebar.markdown("---")
            st.sidebar.title("Refresh Settings")
            auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=True)
            
            if auto_refresh:
                refresh_interval = st.sidebar.slider(
                    "Refresh Interval (seconds)",
                    min_value=5,
                    max_value=60,
                    value=30
                )
                st.sidebar.info(f"Data will refresh every {refresh_interval} seconds")
                
                # Handle auto-refresh
                if auto_refresh:
                    time.sleep(refresh_interval)
                    st.rerun()
            
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error("An unexpected error occurred. Please try again later.")

if __name__ == "__main__":
    main() 