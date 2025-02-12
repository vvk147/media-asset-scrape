import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
from company_data_comparison import analyze_company_data, get_api_usage_stats, set_api_keys, ExaSearcher, ScrapingAntScraper, MediaAsset
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
        # Initialize with empty values in deployment
        st.session_state["SCRAPINGANT_API_KEY"] = ""
        st.session_state["EXA_API_KEY"] = ""
    else:
        # In local development, use .env file
        load_dotenv()
        logger.info("Running locally, loaded environment from .env file")
        st.session_state["env_loaded"] = True
        # Initialize with values from .env
        st.session_state["SCRAPINGANT_API_KEY"] = os.getenv("SCRAPINGANT_API_KEY", "")
        st.session_state["EXA_API_KEY"] = os.getenv("EXA_API_KEY", "")

def initialize_session_state():
    """Initialize session state variables"""
    # Initialize persistent session variables if not already present
    if "is_authenticated" not in st.session_state:
        st.session_state["is_authenticated"] = False
    
    # Initialize API keys from session state if they exist, otherwise from environment
    if "SCRAPINGANT_API_KEY" not in st.session_state:
        st.session_state["SCRAPINGANT_API_KEY"] = os.getenv("SCRAPINGANT_API_KEY", "")
    if "EXA_API_KEY" not in st.session_state:
        st.session_state["EXA_API_KEY"] = os.getenv("EXA_API_KEY", "")
    
    # Add session expiry check
    if "session_start" not in st.session_state:
        st.session_state["session_start"] = datetime.now()
    
    # Initialize filter states
    if "filter_states" not in st.session_state:
        st.session_state.filter_states = {}

def clear_filters():
    """Clear all filter states from session"""
    if "filter_states" in st.session_state:
        st.session_state.filter_states = {}
    # Force rerun to update UI
    st.rerun()

# Initialize session state
initialize_session_state()

# Add sidebar navigation and API key management
st.sidebar.title("Settings")

# Session Information
with st.sidebar.expander("Session Info", expanded=False):
    st.write(f"Session started: {st.session_state.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Session Data"):
            # Keep only essential session state
            for key in list(st.session_state.keys()):
                if key not in ["env_loaded", "session_start"]:
                    del st.session_state[key]
            st.success("Session data cleared! Please refresh the page.")
            st.rerun()
    with col2:
        if st.button("Reset Filters"):
            clear_filters()
            st.success("Filters reset successfully!")

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
    # Generate a unique identifier for this display instance
    display_id = datetime.now().strftime('%Y%m%d%H%M%S')
    
    if show_header:
        st.header("Company Analysis Results")
    
    # Create tabs for different sections
    overview_tab, timeline_tab, market_tab, team_tab, financials_tab, media_tab = st.tabs([
        "Company Overview", "Timeline", "Market & Competition", "Team & Leadership",
        "Financial & Growth", "Media & Documents"
    ])
    
    # Company Overview Tab
    with overview_tab:
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
            
            # Add filters for product categories with session state
            categories = list(set(
                product["category"] for product in company_data["product_lines"]
                if product.get("category")
            ))
            if categories:
                filter_key = f"product_category_filter_{display_id}"
                if filter_key not in st.session_state.filter_states:
                    st.session_state.filter_states[filter_key] = categories
                
                selected_categories = st.multiselect(
                    "Filter by category",
                    options=categories,
                    default=st.session_state.filter_states[filter_key],
                    key=filter_key
                )
                st.session_state.filter_states[filter_key] = selected_categories
            
            # Filter products by selected categories
            filtered_products = [
                product for product in company_data["product_lines"]
                if not categories or not selected_categories or 
                product.get("category") in selected_categories
            ]
            
            # Display products in a grid
            cols = st.columns(3)
            for i, product in enumerate(filtered_products):
                with cols[i % 3]:
                    # Create a card-like container
                    with st.container():
                        # Display product image if available
                        if product.get("image_url"):
                            st.image(product["image_url"], use_column_width=True)
                        
                        # Product name as header
                        st.markdown(f"### {product['name']}")
                        
                        # Category badge if available
                        if product.get("category"):
                            st.markdown(
                                f"""
                                <div style="
                                    display: inline-block;
                                    padding: 0.2em 0.6em;
                                    border-radius: 0.5em;
                                    background-color: #f0f2f6;
                                    margin-bottom: 0.5em;
                                ">
                                    {product['category']}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        # Description
                        if product.get("description"):
                            st.write(product["description"])
                        
                        # Features as bullet points
                        if product.get("features"):
                            st.markdown("**Key Features:**")
                            for feature in product["features"]:
                                st.markdown(f"- {feature}")
                        
                        # Product URL as button
                        if product.get("product_url"):
                            st.markdown(
                                f"""
                                <a href="{product['product_url']}" target="_blank" style="
                                    display: inline-block;
                                    padding: 0.5em 1em;
                                    background-color: #FF4B4B;
                                    color: white;
                                    text-decoration: none;
                                    border-radius: 0.5em;
                                    margin-top: 0.5em;
                                ">
                                    Learn More
                                </a>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        # Add some spacing between products
                        st.markdown("---")
        else:
            st.info("No product lines available")
        
        # Social Media Links
        if company_data["social_links"]:
            st.subheader("Social Media Presence")
            for link in company_data["social_links"]:
                st.write(f"- [{link}]({link})")
    
    # Timeline Tab
    with timeline_tab:
        st.subheader("Company Timeline")
        if company_data.get("timeline_events"):
            # Create a timeline visualization
            events = company_data["timeline_events"]
            
            # Group events by year
            events_by_year = {}
            for event in events:
                year = event['date']
                if year not in events_by_year:
                    events_by_year[year] = []
                events_by_year[year].append(event)
            
            # Sort years
            sorted_years = sorted(events_by_year.keys())
            
            # Display timeline
            for year in sorted_years:
                with st.expander(f"📅 {year}", expanded=True):
                    for event in events_by_year[year]:
                        st.markdown(f"### {event['title']}")
                        if event['description']:
                            st.write(event['description'])
                        if event['source_url']:
                            st.markdown(f"[Source]({event['source_url']})")
                        st.markdown("---")
            
            # Add filters
            st.sidebar.markdown("### Timeline Filters")
            event_types = list(set(event['type'] for event in events))
            selected_types = st.sidebar.multiselect(
                "Filter by event type",
                options=event_types,
                default=event_types,
                key=f"timeline_type_filter_{display_id}"
            )
            
            # Add date range filter
            years = [int(year) for year in sorted_years]
            if years:
                year_range = st.sidebar.slider(
                    "Year range",
                    min_value=min(years),
                    max_value=max(years),
                    value=(min(years), max(years)),
                    key=f"timeline_year_range_{display_id}"
                )
        else:
            st.info("No timeline events available")
    
    # Market & Competition Tab
    with market_tab:
        # Market Position
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Market Position")
            if company_data.get("market_position"):
                for key, value in company_data["market_position"].items():
                    if key == "advantages":
                        st.write("**Competitive Advantages:**")
                        for adv in value:
                            st.write(f"- {adv}")
                    else:
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            else:
                st.info("No market position data available")
        
        with col2:
            st.subheader("Competitors")
            if company_data.get("competitors"):
                # Add sorting options
                sort_options = {
                    "Relevance Score": "relevance_score",
                    "Name": "name"
                }
                sort_by = st.selectbox(
                    "Sort by",
                    list(sort_options.keys()),
                    key=f"competitor_sort_{display_id}"
                )
                
                # Sort competitors
                competitors = sorted(
                    company_data["competitors"],
                    key=lambda x: x[sort_options[sort_by]],
                    reverse=sort_by == "Relevance Score"
                )
                
                # Display competitors
                for competitor in competitors:
                    with st.expander(
                        f"{competitor['name']} (Score: {competitor['relevance_score']})",
                        expanded=False
                    ):
                        # Description
                        if competitor.get("description"):
                            st.write(competitor["description"])
                        
                        # Comparison Points
                        if competitor.get("comparison_points"):
                            st.markdown("**Key Differentiators:**")
                            for point in competitor["comparison_points"]:
                                st.markdown(f"- {point}")
                        
                        # Market Overlap
                        if competitor.get("market_overlap"):
                            st.markdown("**Market Overlap:**")
                            for overlap in competitor["market_overlap"]:
                                st.markdown(f"- {overlap}")
                        
                        # Source Link
                        if competitor.get("source_url"):
                            st.markdown(
                                f"""
                                <a href="{competitor['source_url']}" target="_blank" style="
                                    display: inline-block;
                                    padding: 0.3em 0.6em;
                                    background-color: #f0f2f6;
                                    color: #262730;
                                    text-decoration: none;
                                    border-radius: 0.3em;
                                    margin-top: 0.5em;
                                    font-size: 0.9em;
                                ">
                                    Source
                                </a>
                                """,
                                unsafe_allow_html=True
                            )
            else:
                st.info("No competitor data available")
        
        # Benchmarking Section
        st.markdown("---")
        st.subheader("Market Benchmarking")
        
        if company_data.get("benchmarking_data"):
            benchmarking = company_data["benchmarking_data"]
            
            # Metrics Comparison
            if benchmarking.get("metrics_comparison"):
                st.markdown("### Metrics Comparison")
                
                # Create a DataFrame for the comparison
                metrics_data = []
                for metric, data in benchmarking["metrics_comparison"].items():
                    row = {
                        "Metric": metric.replace("_", " ").title(),
                        "Company": data.get("company", "N/A"),
                        "Industry Avg": data.get("industry_avg", "N/A")
                    }
                    # Add competitor values
                    for comp_name, comp_value in data.get("competitors", {}).items():
                        row[comp_name] = comp_value
                    metrics_data.append(row)
                
                if metrics_data:
                    df = pd.DataFrame(metrics_data)
                    st.dataframe(df.style.highlight_max(axis=1, color='#90EE90'))
            
            # Strengths and Weaknesses
            col1, col2 = st.columns(2)
            with col1:
                if benchmarking.get("strength_weakness", {}).get("strengths"):
                    st.markdown("### Strengths")
                    for strength in benchmarking["strength_weakness"]["strengths"]:
                        st.markdown(f"✅ {strength.replace('_', ' ').title()}")
            
            with col2:
                if benchmarking.get("strength_weakness", {}).get("weaknesses"):
                    st.markdown("### Areas for Improvement")
                    for weakness in benchmarking["strength_weakness"]["weaknesses"]:
                        st.markdown(f"⚠️ {weakness.replace('_', ' ').title()}")
            
            # Market Trends
            if benchmarking.get("market_trends"):
                st.markdown("### Market Trends")
                for trend in benchmarking["market_trends"]:
                    with st.expander(trend["trend"][:100] + "..."):
                        st.write(trend["trend"])
                        if trend.get("source"):
                            st.markdown(f"[Source]({trend['source']})")
        else:
            st.info("No benchmarking data available")
        
        # Market Positioning Section
        st.markdown("---")
        st.subheader("Market Positioning Analysis")
        
        if company_data.get("positioning_analysis"):
            positioning = company_data["positioning_analysis"]
            
            # Market Segments
            if positioning.get("market_segments"):
                st.markdown("### Target Market Segments")
                for segment in positioning["market_segments"]:
                    st.markdown(f"🎯 {segment}")
            
            # Positioning Map
            if positioning.get("positioning_map"):
                st.markdown("### Competitive Positioning Map")
                pos_map = positioning["positioning_map"]
                
                if pos_map.get("dimensions") and pos_map.get("company_position"):
                    # Create scatter plot
                    fig = go.Figure()
                    
                    # Add company position
                    fig.add_trace(go.Scatter(
                        x=[pos_map["company_position"][pos_map["dimensions"][0]]],
                        y=[pos_map["company_position"][pos_map["dimensions"][1]]],
                        mode='markers+text',
                        name=company_data["name"],
                        text=[company_data["name"]],
                        marker=dict(size=15, color='#FF4B4B'),
                        textposition="top center"
                    ))
                    
                    # Add competitor positions
                    for comp_name, comp_pos in pos_map.get("competitor_positions", {}).items():
                        fig.add_trace(go.Scatter(
                            x=[comp_pos[pos_map["dimensions"][0]]],
                            y=[comp_pos[pos_map["dimensions"][1]]],
                            mode='markers+text',
                            name=comp_name,
                            text=[comp_name],
                            marker=dict(size=10),
                            textposition="top center"
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        title="Competitive Positioning",
                        xaxis_title=pos_map["dimensions"][0].title(),
                        yaxis_title=pos_map["dimensions"][1].title(),
                        showlegend=True,
                        width=800,
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Differentiation Factors
            if positioning.get("differentiation_factors"):
                st.markdown("### Key Differentiation Factors")
                for factor in positioning["differentiation_factors"]:
                    st.markdown(f"🔑 {factor}")
            
            # Target Audience
            if positioning.get("target_audience"):
                st.markdown("### Target Audience Analysis")
                for audience in positioning["target_audience"]:
                    with st.expander(audience["segment"][:100] + "..."):
                        st.write(audience["segment"])
                        if audience.get("source"):
                            st.markdown(f"[Source]({audience['source']})")
            
            # Growth Opportunities
            if positioning.get("growth_opportunities"):
                st.markdown("### Growth Opportunities")
                for opportunity in positioning["growth_opportunities"]:
                    with st.expander(opportunity["opportunity"][:100] + "..."):
                        st.write(opportunity["opportunity"])
                        if opportunity.get("source"):
                            st.markdown(f"[Source]({opportunity['source']})")
        else:
            st.info("No positioning analysis available")
    
    # Team & Leadership Tab
    with team_tab:
        st.subheader("Executive Team")
        if company_data.get("executives"):
            cols = st.columns(3)
            for i, executive in enumerate(company_data["executives"]):
                with cols[i % 3]:
                    st.markdown(f"### {executive['name']}")
                    st.write(f"**Role:** {executive['title']}")
                    if executive['bio']:
                        st.write(executive['bio'])
                    if executive['linkedin']:
                        st.write(f"[LinkedIn Profile]({executive['linkedin']})")
        else:
            st.info("No executive team data available")
    
    # Financial & Growth Tab
    with financials_tab:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Financial Metrics")
            if company_data.get("financial_metrics"):
                for metric, value in company_data["financial_metrics"].items():
                    st.metric(
                        label=metric.replace('_', ' ').title(),
                        value=value
                    )
            else:
                st.info("No financial metrics available")
        
        with col2:
            st.subheader("Funding History")
            if company_data.get("funding_rounds"):
                for round_info in company_data["funding_rounds"]:
                    with st.expander(f"{round_info['round_type']} - {round_info['date']}"):
                        st.write(f"**Amount:** {round_info['amount']}")
                        if round_info['investors']:
                            st.write("**Investors:**")
                            for investor in round_info['investors']:
                                st.write(f"- {investor}")
            else:
                st.info("No funding history available")
        
        # Growth Indicators
        st.subheader("Growth Indicators")
        if company_data.get("growth_indicators"):
            cols = st.columns(3)
            for i, indicator in enumerate(company_data["growth_indicators"]):
                with cols[i % 3]:
                    st.metric(
                        label=f"{indicator['metric']} ({indicator['period']})",
                        value=indicator['value']
                    )
        else:
            st.info("No growth indicators available")
        
        # Achievements
        st.subheader("Achievements & Milestones")
        if company_data.get("achievements"):
            for achievement in company_data["achievements"]:
                with st.expander(f"{achievement['title']} ({achievement['date']})"):
                    st.write(achievement['description'])
        else:
            st.info("No achievements data available")
    
    # Media & Documents Tab
    with media_tab:
        # Videos Section
        st.subheader("Videos")
        if company_data.get("videos"):
            # Group videos by platform with unique key
            videos_by_platform = {}
            for video in company_data["videos"]:
                platform = video.get('platform', 'other')
                if platform not in videos_by_platform:
                    videos_by_platform[platform] = []
                videos_by_platform[platform].append(video)
            
            # Create tabs for different video platforms
            if len(videos_by_platform) > 1:
                platform_tabs = st.tabs(list(videos_by_platform.keys()))
                for tab_idx, (tab, (platform, videos)) in enumerate(zip(platform_tabs, videos_by_platform.items())):
                    with tab:
                        for vid_idx, video in enumerate(videos):
                            with st.expander(
                                video['title'] or "Untitled Video",
                                key=f"video_expander_{display_id}_{tab_idx}_{vid_idx}"
                            ):
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    if video['type'] in ['youtube', 'vimeo']:
                                        st.video(video['url'])
                                    else:
                                        st.write(f"[Watch Video]({video['url']})")
                                    if video['description']:
                                        st.write(video['description'])
                                with col2:
                                    if video['thumbnail']:
                                        st.image(video['thumbnail'], use_column_width=True)
                                    if video['duration']:
                                        st.write(f"**Duration:** {video['duration']}")
            else:
                # If only one platform, show videos directly
                for video in company_data["videos"]:
                    with st.expander(video['title'] or "Untitled Video"):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            if video['type'] in ['youtube', 'vimeo']:
                                st.video(video['url'])
                            else:
                                st.write(f"[Watch Video]({video['url']})")
                            if video['description']:
                                st.write(video['description'])
                        with col2:
                            if video['thumbnail']:
                                st.image(video['thumbnail'], use_column_width=True)
                            if video['duration']:
                                st.write(f"**Duration:** {video['duration']}")
        else:
            st.info("No videos available")
        
        st.markdown("---")
        
        # Documents Section
        st.subheader("Documents")
        if company_data.get("documents"):
            # Group documents by type
            docs_by_type = {}
            for doc in company_data["documents"]:
                doc_type = doc['format'].upper()
                if doc_type not in docs_by_type:
                    docs_by_type[doc_type] = []
                docs_by_type[doc_type].append(doc)
            
            # Create columns for different document types
            cols = st.columns(len(docs_by_type) or 1)
            for col, (doc_type, docs) in zip(cols, docs_by_type.items()):
                with col:
                    st.markdown(f"**{doc_type} Files**")
                    for doc in docs:
                        icon = {
                            'PDF': '📄',
                            'DOC': '📝',
                            'DOCX': '📝',
                            'PPT': '📊',
                            'PPTX': '📊',
                            'XLS': '📈',
                            'XLSX': '📈'
                        }.get(doc_type, '📎')
                        st.write(f"{icon} [{doc['title']}]({doc['url']})")
        else:
            st.info("No documents available")
        
        st.markdown("---")
        
        # Media Assets Section
        st.subheader("Media Assets")
        if company_data["media_assets"]:
            # Add filters for media assets with session state
            media_types = list(set(
                asset.type if hasattr(asset, 'type') else asset["type"]
                for asset in company_data["media_assets"]
            ))
            
            media_filter_key = f"media_type_filter_{display_id}"
            if media_filter_key not in st.session_state.filter_states:
                st.session_state.filter_states[media_filter_key] = media_types
            
            selected_types = st.multiselect(
                "Filter by media type",
                media_types,
                default=st.session_state.filter_states[media_filter_key],
                key=media_filter_key
            )
            st.session_state.filter_states[media_filter_key] = selected_types
            
            # Add relevance filters with session state
            relevance_filters = {
                "logo": "Company Logo",
                "product": "Product Images",
                "team": "Team Photos",
                "office": "Office/Location",
                "banner": "Banners/Headers",
                "icon": "Icons/Symbols"
            }
            
            category_filter_key = f"content_category_filter_{display_id}"
            if category_filter_key not in st.session_state.filter_states:
                st.session_state.filter_states[category_filter_key] = [
                    "Company Logo", "Product Images", "Team Photos"
                ]
            
            selected_categories = st.multiselect(
                "Filter by content category",
                options=list(relevance_filters.values()),
                default=st.session_state.filter_states[category_filter_key],
                key=category_filter_key
            )
            st.session_state.filter_states[category_filter_key] = selected_categories
            
            # Function to determine asset relevance
            def is_relevant_asset(asset):
                # Get attributes safely whether it's a dict or MediaAsset
                alt = asset.alt if hasattr(asset, 'alt') else asset.get('alt', '')
                url = asset.url if hasattr(asset, 'url') else asset.get('url', '')
                metadata = asset.metadata if hasattr(asset, 'metadata') else asset.get('metadata', {})
                
                if not alt and not url:
                    return False
                    
                # Skip tiny images (likely icons, spacers, etc.)
                if metadata:
                    width = metadata.get("width")
                    height = metadata.get("height")
                    if width and height and (int(width) < 100 or int(height) < 100):
                        return False
                
                # Check if asset matches selected categories
                asset_text = (alt + " " + url).lower()
                return any(
                    category.lower().replace(" ", "") in asset_text.replace(" ", "")
                    for category in selected_categories
                )
            
            # Filter and display media assets
            filtered_assets = [
                asset for asset in company_data["media_assets"]
                if (asset.type if hasattr(asset, 'type') else asset["type"]) in selected_types 
                and is_relevant_asset(asset)
            ]
            
            if filtered_assets:
                # Create a grid layout
                cols = st.columns(3)
                for i, asset in enumerate(filtered_assets):
                    with cols[i % 3]:
                        try:
                            # Get attributes safely
                            asset_type = asset.type if hasattr(asset, 'type') else asset["type"]
                            asset_url = asset.url if hasattr(asset, 'url') else asset["url"]
                            asset_alt = asset.alt if hasattr(asset, 'alt') else asset.get("alt", "")
                            asset_metadata = asset.metadata if hasattr(asset, 'metadata') else asset.get("metadata", {})
                            
                            if asset_type == "image":
                                caption = asset_alt if asset_alt else "No caption"
                                # Add lightbox effect for images
                                st.markdown(
                                    f"""
                                    <style>
                                        .media-asset_{display_id}_{i} img {{
                                            cursor: pointer;
                                            transition: transform 0.3s ease;
                                        }}
                                        .media-asset_{display_id}_{i} img:hover {{
                                            transform: scale(1.05);
                                        }}
                                    </style>
                                    <div class="media-asset_{display_id}_{i}">
                                        <img src="{asset_url}" alt="{caption}" style="width: 100%">
                                        <figcaption>{caption}</figcaption>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                # Add metadata if available
                                if asset_metadata:
                                    with st.expander("Image Details"):
                                        for key, value in asset_metadata.items():
                                            if value:
                                                st.write(f"**{key.title()}:** {value}")
                        except Exception as e:
                            logger.warning(f"Failed to load media asset: {str(e)}")
                            st.warning("⚠️ Media asset could not be loaded")
            else:
                st.info("No relevant media assets found for selected filters")
        else:
            st.info("No media assets available")

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
                    "media_assets": [
                        MediaAsset(
                            type=asset["type"],
                            url=asset["url"],
                            alt=asset.get("alt", ""),
                            metadata=asset.get("metadata", None)
                        )
                        for asset in eval(full_row["media_assets"]) if pd.notna(full_row["media_assets"])
                    ],
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

def display_api_usage():
    """Display API usage analytics"""
    usage_stats = get_api_usage_stats()
    
    if usage_stats["success"]:
        st.header("📊 API Usage Analytics")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Requests", usage_stats["stats"]["total_requests"])
        with col2:
            st.metric("Total Credits Used", usage_stats["stats"]["total_credits_used"])
        with col3:
            st.metric("Total Cost", f"${usage_stats['stats']['total_cost']:.2f}")
        with col4:
            st.metric("Success Rate", f"{usage_stats['stats']['success_rate']:.1f}%")
        
        # Usage over time
        st.subheader("Usage Over Time")
        
        # Convert daily stats to DataFrame
        usage_df = pd.DataFrame(
            list(usage_stats["stats"]["usage_by_day"].items()),
            columns=["date", "requests"]
        )
        cost_df = pd.DataFrame(
            list(usage_stats["stats"]["cost_by_day"].items()),
            columns=["date", "cost"]
        )
        
        # Plot requests over time
        fig = px.line(
            usage_df,
            x="date",
            y="requests",
            title="Daily API Requests",
            labels={"date": "Date", "requests": "Number of Requests"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot costs over time
        fig = px.line(
            cost_df,
            x="date",
            y="cost",
            title="Daily API Costs",
            labels={"date": "Date", "cost": "Cost ($)"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Usage breakdown
        st.subheader("Usage Breakdown")
        if os.path.exists("api_usage.csv"):
            # Read CSV with proper date handling
            df = pd.read_csv("api_usage.csv")
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Filter controls
            col1, col2 = st.columns(2)
            with col1:
                try:
                    default_start = df["timestamp"].min().date()
                    default_end = df["timestamp"].max().date()
                    date_range = st.date_input(
                        "Date Range",
                        value=(default_start, default_end),
                        min_value=default_start,
                        max_value=default_end
                    )
                    
                    # Handle both single date and date range
                    if isinstance(date_range, tuple):
                        start_date, end_date = date_range
                    else:
                        start_date = end_date = date_range
                except Exception as e:
                    logger.error(f"Error setting date filter: {str(e)}")
                    start_date = df["timestamp"].min().date()
                    end_date = df["timestamp"].max().date()
            
            with col2:
                api_filter = st.multiselect(
                    "Filter by API",
                    options=df["api_name"].unique(),
                    default=df["api_name"].unique()
                )
            
            # Apply filters with safe date handling
            mask = (
                (df["timestamp"].dt.date >= start_date) &
                (df["timestamp"].dt.date <= end_date) &
                (df["api_name"].isin(api_filter))
            )
            filtered_df = df[mask]
            
            # Display detailed table
            st.dataframe(
                filtered_df.style.format({
                    "timestamp": lambda x: pd.to_datetime(x).strftime("%Y-%m-%d %H:%M:%S"),
                    "cost": "${:.4f}",
                    "response_time": "{:.2f} s"
                }),
                hide_index=True
            )
            
            # Prepare CSV for download
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
                if key not in ["env_loaded", "SCRAPINGANT_API_KEY", "EXA_API_KEY", "current_page", "filter_states"]:
                    del st.session_state[key]
            st.session_state.current_page = page
        
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
                display_api_usage()
            
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