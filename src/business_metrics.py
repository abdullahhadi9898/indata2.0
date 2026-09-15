"""
Phase 4: Business Logic & Feature Engineering
Transforms NLP scores into executive metrics that founders care about.
Calculates Churn Risk, Theme Velocity, and extracts critical n-grams.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from collections import Counter
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import json
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnAnalytics:
    """
    Business metrics engine for churn analysis.
    
    Converts sentiment scores into actionable executive metrics:
    - Churn Risk Score (0-100)
    - Theme Velocity (month-over-month change)
    - Critical N-Gram Extraction
    """
    
    def __init__(self):
        """Initialize analytics engine."""
        
        # Churn risk weights (based on business impact)
        self.churn_weights = {
            'billing_subscription': 0.40,  # Highest churn driver
            'customer_service': 0.30,       # Second highest
            'shipping_logistics': 0.20,     # Medium impact
            'packaging': 0.07,              # Lower impact
            'product_quality': 0.03         # Lowest (usually discovered pre-purchase)
        }
        
        # Negative sentiment thresholds
        self.neg_threshold = -0.3
    
    def calculate_churn_risk_score(self, row: pd.Series) -> float:
        """
        Calculate churn risk score (0-100) for a single review.
        
        Formula: Weighted sum of negative sentiment in high-risk aspects.
        
        Args:
            row: DataFrame row with sentiment columns
            
        Returns:
            Churn risk score (0-100)
        """
        risk_score = 0.0
        
        for aspect, weight in self.churn_weights.items():
            sent_col = f'sent_{aspect}'
            if sent_col in row:
                sentiment = row[sent_col]
                
                # Only count negative sentiment
                if sentiment < self.neg_threshold:
                    # Normalize to 0-1 scale (negative only)
                    neg_intensity = abs(sentiment) / 1.0
                    risk_score += weight * neg_intensity * 100
        
        return min(round(risk_score, 1), 100.0)
    
    def add_churn_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add churn risk scores to entire DataFrame.
        
        Args:
            df: Input DataFrame with sentiment columns
            
        Returns:
            DataFrame with added 'churn_risk_score' column
        """
        logger.info("Calculating Churn Risk Scores...")
        
        df_result = df.copy()
        df_result['churn_risk_score'] = df_result.apply(
            self.calculate_churn_risk_score, axis=1
        )
        
        # Categorize risk levels
        def risk_category(score):
            if score >= 70:
                return 'CRITICAL'
            elif score >= 50:
                return 'HIGH'
            elif score >= 30:
                return 'MEDIUM'
            else:
                return 'LOW'
        
        df_result['risk_category'] = df_result['churn_risk_score'].apply(risk_category)
        
        # Log summary
        critical_pct = (df_result['risk_category'] == 'CRITICAL').mean() * 100
        high_pct = (df_result['risk_category'] == 'HIGH').mean() * 100
        
        logger.info(f"Churn Risk Distribution:")
        logger.info(f"  CRITICAL (70+): {critical_pct:.1f}%")
        logger.info(f"  HIGH (50-69): {high_pct:.1f}%")
        
        return df_result
    
    def calculate_theme_velocity(self, df: pd.DataFrame, 
                                  date_column: str = 'date') -> pd.DataFrame:
        """
        Calculate month-over-month change in complaint frequency.
        
        Identifies emerging crises by comparing last 30 days vs previous 30 days.
        
        Args:
            df: Input DataFrame with dates and aspects
            date_column: Name of date column
            
        Returns:
            DataFrame with theme velocity metrics
        """
        logger.info("Calculating Theme Velocity...")
        
        df_result = df.copy()
        df_result[date_column] = pd.to_datetime(df_result[date_column])
        
        # Get date ranges
        max_date = df_result[date_column].max()
        recent_start = max_date - timedelta(days=30)
        previous_start = recent_start - timedelta(days=30)
        
        # Filter by date ranges
        recent = df_result[df_result[date_column] >= recent_start]
        previous = df_result[(df_result[date_column] >= previous_start) & 
                             (df_result[date_column] < recent_start)]
        
        # Calculate complaint rates per aspect
        aspects = ['product_quality', 'shipping_logistics', 'packaging', 
                   'customer_service', 'billing_subscription']
        
        velocity_data = []
        
        for aspect in aspects:
            sent_col = f'sent_{aspect}'
            
            # Count negative mentions (sentiment < threshold)
            recent_neg = (recent[sent_col] < self.neg_threshold).sum()
            previous_neg = (previous[sent_col] < self.neg_threshold).sum()
            
            # Calculate rates
            recent_total = len(recent)
            previous_total = len(previous)
            
            recent_rate = recent_neg / recent_total if recent_total > 0 else 0
            previous_rate = previous_neg / previous_total if previous_total > 0 else 0
            
            # Calculate velocity (% change)
            if previous_rate > 0:
                velocity = ((recent_rate - previous_rate) / previous_rate) * 100
            else:
                velocity = 100 if recent_rate > 0 else 0
            
            velocity_data.append({
                'aspect': aspect,
                'recent_negative_count': int(recent_neg),
                'previous_negative_count': int(previous_neg),
                'recent_rate': round(recent_rate, 4),
                'previous_rate': round(previous_rate, 4),
                'velocity_percent': round(velocity, 1),
                'trend': 'SPIKING' if velocity > 20 else ('DECLINING' if velocity < -20 else 'STABLE')
            })
        
        df_velocity = pd.DataFrame(velocity_data)
        
        # Log spiking themes
        spiking = df_velocity[df_velocity['trend'] == 'SPIKING']
        if len(spiking) > 0:
            logger.warning(f"⚠️  SPIKING complaints detected:")
            for _, row in spiking.iterrows():
                logger.warning(f"  {row['aspect']}: +{row['velocity_percent']}%")
        
        return df_velocity
    
    def extract_critical_ngrams(self, df: pd.DataFrame, 
                                 text_column: str = 'cleaned_text',
                                 top_n: int = 20) -> List[Dict]:
        """
        Extract top n-grams from high-churn-risk reviews.
        
        Identifies exact phrases driving negative sentiment.
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            top_n: Number of top phrases to return
            
        Returns:
            List of dictionaries with phrase and count
        """
        logger.info(f"Extracting top {top_n} critical n-grams...")
        
        # Filter to high-risk reviews
        high_risk = df[df['churn_risk_score'] >= 50]
        
        if len(high_risk) == 0:
            logger.warning("No high-risk reviews found for n-gram extraction")
            return []
        
        # Combine all text
        all_text = ' '.join(high_risk[text_column].astype(str))
        
        # Tokenize
        try:
            tokens = word_tokenize(all_text.lower())
        except Exception:
            tokens = all_text.lower().split()
        
        # Remove stopwords and short words
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'this', 'that', 'it', 'as', 'so', 'if', 'then', 'than', 'too', 'very'
        }
        
        tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
        
        # Extract bi-grams and tri-grams
        bi_grams = list(ngrams(tokens, 2))
        tri_grams = list(ngrams(tokens, 3))
        
        # Count frequencies
        bi_counts = Counter([' '.join(bg) for bg in bi_grams])
        tri_counts = Counter([' '.join(tg) for tg in tri_grams])
        
        # Combine and get top phrases
        all_phrases = bi_counts + tri_counts
        top_phrases = all_phrases.most_common(top_n)
        
        result = [{'phrase': phrase, 'count': count} for phrase, count in top_phrases]
        
        logger.info(f"Top critical phrases identified")
        
        return result
    
    def generate_executive_summary(self, df: pd.DataFrame, 
                                    velocity_df: pd.DataFrame) -> Dict:
        """
        Generate executive summary JSON for dashboard.
        
        Args:
            df: Full DataFrame with churn scores
            velocity_df: Theme velocity DataFrame
            
        Returns:
            Dictionary with key metrics
        """
        logger.info("Generating executive summary...")
        
        # Calculate key metrics
        total_reviews = len(df)
        avg_rating = df['rating'].mean()
        
        # Average sentiment by aspect
        aspects = ['product_quality', 'shipping_logistics', 'packaging', 
                   'customer_service', 'billing_subscription']
        
        avg_sentiments = {}
        for aspect in aspects:
            sent_col = f'sent_{aspect}'
            avg_sentiments[aspect] = round(df[sent_col].mean(), 3)
        
        # Primary churn driver
        neg_counts = {}
        for aspect in aspects:
            sent_col = f'sent_{aspect}'
            neg_counts[aspect] = (df[sent_col] < self.neg_threshold).sum()
        
        primary_driver = max(neg_counts.items(), key=lambda x: x[1])[0]
        
        # Churn risk stats
        avg_churn_risk = df['churn_risk_score'].mean()
        high_risk_pct = (df['churn_risk_score'] >= 50).mean() * 100
        
        # Trend direction
        spiking_aspects = velocity_df[velocity_df['trend'] == 'SPIKING']['aspect'].tolist()
        declining_aspects = velocity_df[velocity_df['trend'] == 'DECLINING']['aspect'].tolist()
        
        if len(spiking_aspects) > len(declining_aspects):
            trend_direction = 'DECLINING'
        elif len(declining_aspects) > len(spiking_aspects):
            trend_direction = 'IMPROVING'
        else:
            trend_direction = 'STABLE'
        
        summary = {
            'total_reviews': int(total_reviews),
            'avg_rating': round(avg_rating, 2),
            'avg_sentiments': avg_sentiments,
            'primary_churn_driver': primary_driver,
            'avg_churn_risk': round(avg_churn_risk, 1),
            'high_risk_percentage': round(high_risk_pct, 1),
            'trend_direction': trend_direction,
            'spiking_aspects': spiking_aspects,
            'declining_aspects': declining_aspects,
            'generated_at': datetime.now().isoformat()
        }
        
        return summary


def run_business_analytics(input_path: str = 'data/enriched_reviews.csv',
                            output_path: str = 'data/final_analytics_dataset.csv',
                            summary_path: str = 'data/executive_summary.json') -> Tuple[pd.DataFrame, Dict]:
    """
    Run full business analytics pipeline.
    
    Args:
        input_path: Path to enriched CSV
        output_path: Path to save final dataset
        summary_path: Path to save executive summary JSON
        
    Returns:
        Tuple of (final DataFrame, executive summary dict)
    """
    logger.info(f"Loading enriched data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Initialize analytics engine
    analytics = ChurnAnalytics()
    
    # Add churn risk scores
    df_final = analytics.add_churn_risk_scores(df)
    
    # Calculate theme velocity
    df_velocity = analytics.calculate_theme_velocity(df_final)
    
    # Extract critical n-grams
    critical_phrases = analytics.extract_critical_ngrams(df_final)
    
    # Generate executive summary
    summary = analytics.generate_executive_summary(df_final, df_velocity)
    summary['top_critical_phrases'] = critical_phrases
    summary['theme_velocity'] = df_velocity.to_dict('records')
    
    # Save final dataset
    df_final.to_csv(output_path, index=False)
    logger.info(f"✅ Saved {len(df_final)} reviews to {output_path}")
    
    # Save executive summary
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✅ Saved executive summary to {summary_path}")
    
    # Print summary
    print("\n=== Executive Summary ===")
    print(f"Total Reviews: {summary['total_reviews']}")
    print(f"Average Rating: {summary['avg_rating']}")
    print(f"Primary Churn Driver: {summary['primary_churn_driver']}")
    print(f"High Risk %: {summary['high_risk_percentage']}%")
    print(f"Trend: {summary['trend_direction']}")
    print(f"\nTop Critical Phrases:")
    for phrase in critical_phrases[:5]:
        print(f"  - \"{phrase['phrase']}\" ({phrase['count']} mentions)")
    
    return df_final, summary


if __name__ == "__main__":
    df_final, summary = run_business_analytics()
    print("\n✅ Phase 4 Complete: Business Logic & Feature Engineering finished!")
