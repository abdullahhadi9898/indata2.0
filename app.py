"""
Phase 5: The "Million-Dollar" Streamlit Dashboard
Interactive executive dashboard for e-commerce churn analysis.
Professional UI with consultant-grade visualizations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="InData | Brand Health & Churn Audit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #2C3E50;
        --secondary-color: #3498DB;
        --accent-color: #E74C3C;
        --success-color: #27AE60;
        --warning-color: #F39C12;
        --bg-color: #ECF0F1;
        --card-bg: #FFFFFF;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Card styling */
    .metric-card {
        background: var(--card-bg);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* KPI text */
    .kpi-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: var(--primary-color);
    }
    
    .kpi-label {
        font-size: 0.9rem;
        color: #7F8C8D;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--primary-color);
        margin: 30px 0 15px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid var(--secondary-color);
    }
    
    /* Sidebar styling */
    .sidebar-content {
        background: var(--card-bg);
        padding: 20px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


def load_data():
    """Load analytics dataset and executive summary."""
    data_path = 'data/final_analytics_dataset.csv'
    summary_path = 'data/executive_summary.json'
    
    if not os.path.exists(data_path):
        st.error("❌ Data file not found. Please run the pipeline first: `python src/run_pipeline.py`")
        return None, None
    
    df = pd.read_csv(data_path)
    
    summary = None
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
    
    return df, summary


def render_kpi_cards(df, summary):
    """Render executive KPI cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="kpi-label">Total Reviews</div>
            <div class="kpi-value">{len(df):,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_rating = df['rating'].mean()
        rating_color = "#27AE60" if avg_rating >= 4 else ("#F39C12" if avg_rating >= 3 else "#E74C3C")
        st.markdown(f"""
        <div class="metric-card">
            <div class="kpi-label">Avg Rating</div>
            <div class="kpi-value" style="color: {rating_color}">{avg_rating:.2f}⭐</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        high_risk_pct = (df['churn_risk_score'] >= 50).mean() * 100
        risk_color = "#E74C3C" if high_risk_pct > 30 else ("#F39C12" if high_risk_pct > 15 else "#27AE60")
        st.markdown(f"""
        <div class="metric-card">
            <div class="kpi-label">High Churn Risk</div>
            <div class="kpi-value" style="color: {risk_color}">{high_risk_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if summary and 'primary_churn_driver' in summary:
            driver = summary['primary_churn_driver'].replace('_', ' ').title()
        else:
            driver = "Unknown"
        st.markdown(f"""
        <div class="metric-card">
            <div class="kpi-label">Top Churn Driver</div>
            <div class="kpi-value" style="font-size: 1.5rem; color: #E74C3C">{driver}</div>
        </div>
        """, unsafe_allow_html=True)


def render_root_cause_chart(df):
    """Render bar chart of root causes for 1-star reviews."""
    aspects = ['product_quality', 'shipping_logistics', 'packaging', 
               'customer_service', 'billing_subscription']
    
    # Filter to 1-2 star reviews
    negative_reviews = df[df['rating'] <= 2]
    
    # Count negative sentiment by aspect
    cause_counts = {}
    aspect_names = {
        'product_quality': 'Product Quality',
        'shipping_logistics': 'Shipping/Logistics',
        'packaging': 'Packaging',
        'customer_service': 'Customer Service',
        'billing_subscription': 'Billing/Subscription'
    }
    
    for aspect in aspects:
        sent_col = f'sent_{aspect}'
        # Count reviews with negative sentiment for this aspect
        neg_count = (negative_reviews[sent_col] < -0.3).sum()
        cause_counts[aspect_names[aspect]] = neg_count
    
    # Create DataFrame for plotting
    df_causes = pd.DataFrame({
        'Root Cause': list(cause_counts.keys()),
        'Count': list(cause_counts.values())
    }).sort_values('Count', ascending=False)
    
    # Create Plotly bar chart
    fig = px.bar(
        df_causes,
        x='Root Cause',
        y='Count',
        title="<b>Root Causes of 1-2 Star Reviews</b>",
        color='Count',
        color_continuous_scale='Reds',
        text='Count'
    )
    
    fig.update_traces(textposition='outside', textfont=dict(size=12))
    fig.update_layout(
        height=400,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=12),
        xaxis_title="",
        yaxis_title="Number of Reviews",
        hovermode='x unified'
    )
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='#ECF0F1')
    
    return fig


def render_sentiment_trend_chart(df):
    """Render line chart of sentiment trend by aspect over time."""
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['date'])
    
    # Group by month and aspect
    df_copy['month'] = df_copy['date'].dt.to_period('M').dt.to_timestamp()
    
    aspects = ['product_quality', 'shipping_logistics', 'packaging', 
               'customer_service', 'billing_subscription']
    
    aspect_names = {
        'product_quality': 'Product Quality',
        'shipping_logistics': 'Shipping/Logistics',
        'packaging': 'Packaging',
        'customer_service': 'Customer Service',
        'billing_subscription': 'Billing/Subscription'
    }
    
    # Calculate monthly average sentiment
    monthly_data = []
    for aspect in aspects:
        sent_col = f'sent_{aspect}'
        grouped = df_copy.groupby('month')[sent_col].mean().reset_index()
        grouped['Aspect'] = aspect_names[aspect]
        grouped['Sentiment'] = grouped[sent_col]
        monthly_data.append(grouped[['month', 'Aspect', 'Sentiment']])
    
    df_monthly = pd.concat(monthly_data, ignore_index=True)
    
    # Create Plotly line chart
    fig = px.line(
        df_monthly,
        x='month',
        y='Sentiment',
        color='Aspect',
        title="<b>Sentiment Trend by Aspect Over Time</b>",
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_layout(
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=11),
        xaxis_title="Month",
        yaxis_title="Average Sentiment Score",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(showgrid=False, tickformat="%b %Y")
    fig.update_yaxes(showgrid=True, gridcolor='#ECF0F1', range=[-1, 1])
    
    # Add threshold lines
    fig.add_hline(y=-0.3, line_dash="dash", line_color="#E74C3C", 
                  annotation_text="Negative Threshold", annotation_position="right")
    fig.add_hline(y=0.3, line_dash="dash", line_color="#27AE60",
                  annotation_text="Positive Threshold", annotation_position="right")
    
    return fig


def render_theme_velocity_chart(summary):
    """Render theme velocity metrics."""
    if not summary or 'theme_velocity' not in summary:
        return None
    
    velocity_data = summary['theme_velocity']
    df_velocity = pd.DataFrame(velocity_data)
    
    # Create grouped bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Previous 30 Days',
        x=df_velocity['aspect'],
        y=df_velocity['previous_negative_count'],
        marker_color='#3498DB',
        text=df_velocity['previous_negative_count'],
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        name='Last 30 Days',
        x=df_velocity['aspect'],
        y=df_velocity['recent_negative_count'],
        marker_color=df_velocity['velocity_percent'].apply(
            lambda x: '#E74C3C' if x > 20 else ('#27AE60' if x < -20 else '#F39C12')
        ),
        text=df_velocity['recent_negative_count'],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="<b>Theme Velocity: Complaint Trends (30-Day Comparison)</b>",
        barmode='group',
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=11),
        xaxis_title="Aspect",
        yaxis_title="Number of Negative Mentions",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='#ECF0F1')
    
    return fig


def render_high_risk_quotes(df):
    """Render table of high-risk customer quotes."""
    # Filter to highest risk reviews
    high_risk = df[df['churn_risk_score'] >= 70].copy()
    
    if len(high_risk) == 0:
        st.info("No critical risk reviews found.")
        return
    
    # Sort by risk score
    high_risk = high_risk.sort_values('churn_risk_score', ascending=False)
    
    # Select columns for display
    display_cols = ['date', 'rating', 'review_text', 'churn_risk_score', 'aspect_primary']
    
    # Rename for display
    rename_map = {
        'date': 'Date',
        'rating': 'Rating',
        'review_text': 'Customer Quote',
        'churn_risk_score': 'Risk Score',
        'aspect_primary': 'Primary Issue'
    }
    
    # Show top 10
    top_10 = high_risk.head(10)[display_cols].rename(columns=rename_map)
    
    # Format for display
    st.dataframe(
        top_10.style.format({
            'Risk Score': '{:.1f}',
            'Rating': lambda x: f"{'⭐' * int(x)}"
        }).background_gradient(
            subset=['Risk Score'], cmap='Reds'
        ),
        use_container_width=True,
        height=400
    )


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2C3E50 0%, #3498DB 100%); 
                padding: 30px; border-radius: 15px; margin-bottom: 30px;">
        <h1 style="color: white; margin: 0;">📊 InData Analytics</h1>
        <h2 style="color: #ECF0F1; margin: 10px 0 0 0; font-weight: 300;">
            E-commerce Brand Health & Churn Audit
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df, summary = load_data()
    
    if df is None:
        return
    
    # Sidebar filters
    st.sidebar.markdown("### 🔍 Filters")
    
    # Date range filter
    df['date'] = pd.to_datetime(df['date'])
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min=min_date,
        max=max_date
    )
    
    # Rating filter
    rating_filter = st.sidebar.multiselect(
        "Star Rating",
        options=[1, 2, 3, 4, 5],
        default=[1, 2, 3, 4, 5]
    )
    
    # Aspect filter
    aspect_options = ['All'] + ['product_quality', 'shipping_logistics', 
                                 'packaging', 'customer_service', 'billing_subscription']
    aspect_filter = st.sidebar.selectbox("Primary Aspect", aspect_options)
    
    # Apply filters
    filtered_df = df.copy()
    
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= date_range[0]) & 
            (filtered_df['date'].dt.date <= date_range[1])
        ]
    
    if rating_filter:
        filtered_df = filtered_df[filtered_df['rating'].isin(rating_filter)]
    
    if aspect_filter != 'All':
        filtered_df = filtered_df[filtered_df['aspect_primary'] == aspect_filter]
    
    st.sidebar.markdown(f"**Reviews shown:** {len(filtered_df):,} of {len(df):,}")
    
    # Render KPI cards
    render_kpi_cards(filtered_df, summary)
    
    # Main charts row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(render_root_cause_chart(filtered_df), use_container_width=True)
    
    with col2:
        st.plotly_chart(render_sentiment_trend_chart(filtered_df), use_container_width=True)
    
    # Theme velocity section
    st.markdown('<div class="section-header">🚨 Emerging Issues (Theme Velocity)</div>', 
                unsafe_allow_html=True)
    
    if summary and 'theme_velocity' in summary:
        spiking = [v for v in summary['theme_velocity'] if v['trend'] == 'SPIKING']
        if spiking:
            st.warning(f"**⚠️ Alert:** {len(spiking)} aspect(s) showing spiking complaints")
            for issue in spiking:
                st.write(f"- **{issue['aspect'].replace('_', ' ').title()}**: +{issue['velocity_percent']}% vs previous 30 days")
    
    st.plotly_chart(render_theme_velocity_chart(summary), use_container_width=True)
    
    # High-risk quotes section
    st.markdown('<div class="section-header">💬 High-Risk Customer Quotes</div>', 
                unsafe_allow_html=True)
    st.write("Top 10 reviews with highest churn risk scores (requires immediate attention)")
    
    render_high_risk_quotes(filtered_df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7F8C8D; padding: 20px;">
        <p><strong>InData Analytics</strong> | Elite Data Science & Business Intelligence</p>
        <p>Built for DTC E-commerce Brands • Skincare • Supplements • Wellness</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
