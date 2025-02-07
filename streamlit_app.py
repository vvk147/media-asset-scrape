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

# Load environment variables from .env file in local development
if os.getenv("DEPLOYMENT_ENV") != "production":
    load_dotenv()

# Ensure the virtual environment is being used
def check_venv():
    base_dir = Path(__file__).parent
    venv_dir = base_dir / ".venv"
    if not venv_dir.exists():
        print("Virtual environment not found. Please create one using:")
        print("python -m venv .venv")
        print("And activate it, then run:")
        print("pip install -r requirements.txt")
        sys.exit(1)

def load_api_keys() -> tuple[Optional[str], Optional[str]]:
    """Load API keys from environment or session state"""
    if os.getenv("DEPLOYMENT_ENV") == "production":
        # In production, always use session state
        return (
            st.session_state.get("SCRAPINGANT_API_KEY"),
            st.session_state.get("EXA_API_KEY")
        )
    else:
        # In local development, prefer environment variables
        scrapingant_key = os.getenv("SCRAPINGANT_API_KEY")
        exa_key = os.getenv("EXA_API_KEY")
        
        # If either key is missing from env, try session state
        if not scrapingant_key or not exa_key:
            return (
                scrapingant_key or st.session_state.get("SCRAPINGANT_API_KEY"),
                exa_key or st.session_state.get("EXA_API_KEY")
            )
        
        # Store env keys in session state for consistency
        if "SCRAPINGANT_API_KEY" not in st.session_state:
            st.session_state["SCRAPINGANT_API_KEY"] = scrapingant_key
        if "EXA_API_KEY" not in st.session_state:
            st.session_state["EXA_API_KEY"] = exa_key
        
        return (scrapingant_key, exa_key)

def save_api_keys(scrapingant_key: str, exa_key: str):
    """Save API keys to session state"""
    st.session_state["SCRAPINGANT_API_KEY"] = scrapingant_key
    st.session_state["EXA_API_KEY"] = exa_key

def initialize_session_state():
    """Initialize session state variables"""
    if "SCRAPINGANT_API_KEY" not in st.session_state:
        st.session_state["SCRAPINGANT_API_KEY"] = None
    if "EXA_API_KEY" not in st.session_state:
        st.session_state["EXA_API_KEY"] = None

# Add environment check at the start
if __name__ == "__main__":
    check_venv()

# Configure page settings
st.set_page_config(
    page_title="Company Data Analysis",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
initialize_session_state()

# Add sidebar navigation and API key management
st.sidebar.title("Settings")

# API Key Management Section
with st.sidebar.expander("API Key Management", expanded=True):
    scrapingant_key, exa_key = load_api_keys()
    
    # Show current source of API keys
    if os.getenv("DEPLOYMENT_ENV") != "production":
        if os.getenv("SCRAPINGANT_API_KEY") and os.getenv("EXA_API_KEY"):
            st.success("✅ Using API keys from .env file")
            st.info("You can override these keys below if needed")
    
    new_scrapingant_key = st.text_input(
        "ScrapingAnt API Key",
        value=scrapingant_key or "",
        type="password",
        help="Enter your ScrapingAnt API key" + 
             (" (currently using from .env)" if os.getenv("SCRAPINGANT_API_KEY") else "")
    )
    
    new_exa_key = st.text_input(
        "Exa.ai API Key",
        value=exa_key or "",
        type="password",
        help="Enter your Exa.ai API key" + 
             (" (currently using from .env)" if os.getenv("EXA_API_KEY") else "")
    )
    
    if st.button("Save API Keys"):
        save_api_keys(new_scrapingant_key, new_exa_key)
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

def display_history():
    """Display historical data from CSV"""
    if os.path.exists("company_data_history.csv"):
        df = pd.read_csv("company_data_history.csv")
        df['scrape_date'] = pd.to_datetime(df['scrape_date'])
        df = df.sort_values('scrape_date', ascending=False)
        
        st.subheader("Analysis History")
        
        # Add filters
        col1, col2 = st.columns(2)
        with col1:
            date_filter = st.date_input(
                "Filter by date",
                value=[df['scrape_date'].min().date(), df['scrape_date'].max().date()]
            )
        with col2:
            domain_filter = st.multiselect(
                "Filter by domain",
                options=df['domain'].unique()
            )
        
        # Apply filters
        mask = (df['scrape_date'].dt.date >= date_filter[0]) & (df['scrape_date'].dt.date <= date_filter[1])
        if domain_filter:
            mask = mask & (df['domain'].isin(domain_filter))
        
        filtered_df = df[mask]
        
        # Display data
        st.dataframe(
            filtered_df.style.format({
                'scrape_date': lambda x: x.strftime('%Y-%m-%d %H:%M:%S')
            }),
            hide_index=True
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name=f"company_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
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
        
        # Convert dictionary to DataFrame for plotting
        usage_df = pd.DataFrame([
            {"date": date, "requests": count}
            for date, count in stats['usage_by_day'].items()
        ])
        usage_df['date'] = pd.to_datetime(usage_df['date'])
        
        # Create line chart
        fig = px.line(
            usage_df, 
            x='date', 
            y='requests',
            title='Daily API Requests',
            labels={'date': 'Date', 'requests': 'Number of Requests'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cost over time
        cost_df = pd.DataFrame([
            {"date": date, "cost": cost}
            for date, cost in stats['cost_by_day'].items()
        ])
        cost_df['date'] = pd.to_datetime(cost_df['date'])
        
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
    # Load API keys
    scrapingant_key, exa_key = load_api_keys()

    if page == "Company Analysis":
        st.title("🏢 Company Data Analysis Tool")
        
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
                            # Create three columns for metrics
                            col1, col2, col3 = st.columns(3)
                            
                            # Display metrics as gauge charts
                            with col1:
                                st.plotly_chart(create_gauge_chart(
                                    results["metrics"]["social_presence_score"],
                                    "Social Presence Score"
                                ), use_container_width=True)
                                
                            with col2:
                                st.plotly_chart(create_gauge_chart(
                                    results["metrics"]["media_richness_score"],
                                    "Media Richness Score"
                                ), use_container_width=True)
                                
                            with col3:
                                st.plotly_chart(create_gauge_chart(
                                    results["metrics"]["information_completeness"],
                                    "Information Completeness"
                                ), use_container_width=True)
                            
                            # Display company details
                            st.subheader("Company Details")
                            company_data = results["company_data"]
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Name:**", company_data["name"])
                                st.write("**Domain:**", company_data["domain"])
                                if company_data["logo_url"]:
                                    st.image(company_data["logo_url"], width=100)
                            
                            with col2:
                                if company_data["description"]:
                                    st.write("**Description:**", company_data["description"])
                            
                            # Display social links
                            if company_data["social_links"]:
                                st.subheader("Social Media Presence")
                                for link in company_data["social_links"]:
                                    st.write(f"- [{link}]({link})")
                            
                            # Display media assets
                            if company_data["media_assets"]:
                                st.subheader("Media Assets")
                                cols = st.columns(3)
                                for i, asset in enumerate(company_data["media_assets"]):
                                    if asset.type == "image":
                                        with cols[i % 3]:
                                            st.image(
                                                asset.url,
                                                caption=asset.alt if asset.alt else "No caption",
                                                use_column_width=True
                                            )
                        else:
                            st.error(f"Analysis failed: {results['error']}")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
            else:
                st.warning("Please enter a company URL")
        
        # Display historical data
        display_history()
    
    else:  # API Analytics page
        display_api_analytics()

if __name__ == "__main__":
    main() 