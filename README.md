# InData - E-commerce Brand Health & Churn Audit

## Overview
A professional Aspect-Based Sentiment Analysis (ABSA) pipeline designed to help DTC e-commerce founders understand **why** customers are churning. This tool ingests messy review data, cleans it, categorizes feedback into 5 core business themes, and outputs an interactive executive dashboard.

## Business Value
- **Identify Churn Drivers**: Pinpoint whether customers leave due to Product Quality, Shipping, Packaging, Customer Service, or Billing issues.
- **Theme Velocity**: Detect emerging crises before they become brand-damaging.
- **Actionable Insights**: Move beyond "4.2 stars" to specific quotes and metrics that drive revenue decisions.

## Project Structure
```
/
├── src/
│   ├── generate_synthetic_data.py    # Phase 1: Realistic synthetic review generator
│   ├── web_scraper.py                # Phase 1: Production-grade Trustpilot/Amazon scraper
│   ├── data_cleaner.py               # Phase 2: Modular cleaning & preprocessing
│   ├── sentiment_engine.py           # Phase 3: Aspect-Based Sentiment Engine
│   ├── business_metrics.py           # Phase 4: Churn risk & feature engineering
│   ├── run_pipeline.py               # Master script to run full pipeline
│   └── __init__.py
├── data/
│   ├── synthetic_reviews.csv         # Generated raw data
│   ├── cleaned_reviews.csv           # Preprocessed data
│   ├── enriched_reviews.csv          # With aspect scores
│   └── final_analytics_dataset.csv   # Ready for dashboard
├── app.py                            # Phase 5: Streamlit Dashboard
├── requirements.txt                  # Dependencies
├── README.md
└── LICENSE
```

## Quick Start

### 1. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run Full Pipeline
```bash
python src/run_pipeline.py
```
This generates synthetic data, cleans it, runs sentiment analysis, and calculates business metrics.

### 3. Launch Dashboard
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

## Core Features

### 5 Business Aspects Analyzed
1. **Product Quality** - Efficacy, texture, scent, breakage
2. **Shipping/Logistics** - Delivery time, carrier issues, lost packages
3. **Packaging** - Damaged boxes, leaks, eco-friendliness
4. **Customer Service** - Response time, helpfulness, rudeness
5. **Billing/Subscription** - Refunds, cancellation difficulty, hidden charges

### Key Metrics
- **Churn Risk Score (0-100)**: Composite metric predicting customer loss likelihood
- **Theme Velocity**: Month-over-month change in complaint frequency
- **Aspect Sentiment Scores**: -1.0 to +1.0 for each business aspect per review

## Tech Stack
- **Python 3.10+**
- **Data**: pandas, numpy
- **NLP**: transformers, spacy, nltk, scikit-learn
- **Visualization**: streamlit, plotly
- **Scraping**: requests, beautifulsoup4, lxml

## Consulting Use Case
This project is designed as a $5,000 consulting deliverable for:
- DTC Skincare brands
- Supplement companies
- Wellness product founders
- Subscription box services

Use this to win clients on Upwork or via cold outreach by demonstrating deep customer insight capabilities.

## License
MIT License - See LICENSE file

---
*Built by InData - Elite Data Science & Business Analytics*
