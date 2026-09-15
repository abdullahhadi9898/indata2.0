"""
Master Pipeline Runner
Executes all 5 phases of the InData Churn Audit pipeline.
Run this script to generate data, clean, analyze, and prepare for dashboard.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_full_pipeline():
    """Execute complete pipeline from data generation to analytics."""
    
    print("=" * 60)
    print("InData - E-commerce Brand Health & Churn Audit")
    print("Full Pipeline Execution")
    print("=" * 60)
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Phase 1: Generate Synthetic Data
    print("\n" + "=" * 60)
    print("PHASE 1: Data Generation")
    print("=" * 60)
    from generate_synthetic_data import generate_synthetic_reviews, save_to_csv
    
    df_raw = generate_synthetic_reviews(2000)
    save_to_csv(df_raw, 'data/synthetic_reviews.csv')
    
    # Phase 2: Clean Data
    print("\n" + "=" * 60)
    print("PHASE 2: Data Cleaning")
    print("=" * 60)
    from data_cleaner import load_and_clean_data
    
    df_clean = load_and_clean_data(
        input_path='data/synthetic_reviews.csv',
        output_path='data/cleaned_reviews.csv'
    )
    
    # Phase 3: Sentiment Analysis
    print("\n" + "=" * 60)
    print("PHASE 3: Aspect-Based Sentiment Analysis")
    print("=" * 60)
    from sentiment_engine import run_sentiment_analysis
    
    df_enriched = run_sentiment_analysis(
        input_path='data/cleaned_reviews.csv',
        output_path='data/enriched_reviews.csv'
    )
    
    # Phase 4: Business Metrics
    print("\n" + "=" * 60)
    print("PHASE 4: Business Logic & Feature Engineering")
    print("=" * 60)
    from business_metrics import run_business_analytics
    
    df_final, summary = run_business_analytics(
        input_path='data/enriched_reviews.csv',
        output_path='data/final_analytics_dataset.csv',
        summary_path='data/executive_summary.json'
    )
    
    # Pipeline Complete
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE")
    print("=" * 60)
    print("\nGenerated Files:")
    print("  1. data/synthetic_reviews.csv (raw)")
    print("  2. data/cleaned_reviews.csv (preprocessed)")
    print("  3. data/enriched_reviews.csv (with aspect scores)")
    print("  4. data/final_analytics_dataset.csv (ready for dashboard)")
    print("  5. data/executive_summary.json (KPIs)")
    print("\nNext Step: Run dashboard with:")
    print("  streamlit run app.py")
    print("=" * 60)
    
    return df_final, summary


if __name__ == "__main__":
    run_full_pipeline()
