"""
Phase 3: Aspect-Based Sentiment Engine (ABSE)
The core value proposition - assigns sentiment scores to 5 business aspects.
Uses hybrid Zero-Shot Classification + Keyword mapping for accuracy and speed.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from transformers import pipeline
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AspectSentimentEngine:
    """
    Aspect-Based Sentiment Analysis Engine.
    
    Analyzes reviews across 5 core business aspects:
    1. Product Quality
    2. Shipping/Logistics
    3. Packaging
    4. Customer Service
    5. Billing/Subscription
    
    Output: Sentiment score (-1.0 to +1.0) for EACH aspect per review.
    """
    
    def __init__(self, model_name: str = "typeform/distilbert-base-uncased-mnli"):
        """
        Initialize sentiment engine with zero-shot classification model.
        
        Args:
            model_name: HuggingFace model for zero-shot classification
        """
        self.aspect_labels = [
            "Product quality and efficacy",
            "Shipping and delivery logistics", 
            "Packaging and product protection",
            "Customer service and support",
            "Billing and subscription management"
        ]
        
        # Aspect keyword mappings (fallback method)
        self.aspect_keywords = {
            'product_quality': [
                'product', 'formula', 'texture', 'scent', 'smell', 'skin', 'face',
                'breakout', 'acne', 'moisturizer', 'serum', 'cream', 'lotion',
                'effective', 'works', 'results', 'ingredient', 'quality'
            ],
            'shipping_logistics': [
                'shipping', 'delivery', 'arrived', 'package', 'carrier', 'fedex',
                'ups', 'dhl', 'mail', 'post', 'tracking', 'delayed', 'late',
                'fast', 'slow', 'week', 'day', 'time', 'arrive'
            ],
            'packaging': [
                'packaging', 'box', 'bottle', 'jar', 'container', 'leak', 'leaked',
                'cracked', 'broken', 'damaged', 'sealed', 'cap', 'pump', 'dropper',
                'wrapped', 'protected', 'eco-friendly', 'recyclable'
            ],
            'customer_service': [
                'service', 'support', 'help', 'response', 'reply', 'email', 'call',
                'phone', 'chat', 'representative', 'agent', 'staff', 'team',
                'rude', 'helpful', 'friendly', 'wait', 'hold', 'transfer'
            ],
            'billing_subscription': [
                'billing', 'charge', 'charged', 'payment', 'refund', 'subscription',
                'cancel', 'cancelled', 'renew', 'auto-ship', 'fee', 'cost', 'price',
                'expensive', 'cheap', 'money', 'card', 'account'
            ]
        }
        
        # Load zero-shot classifier
        logger.info(f"Loading sentiment model: {model_name}")
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=-1,  # CPU (set to 0 for GPU)
                batch_size=8
            )
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            logger.info("Falling back to keyword-only mode")
            self.classifier = None
        
        # Cache for repeated texts
        self.cache = {}
    
    def _keyword_aspect_detection(self, text: str) -> Dict[str, float]:
        """
        Detect aspects using keyword matching.
        
        Returns confidence score (0.0 to 1.0) for each aspect based on keyword density.
        """
        text_lower = text.lower()
        aspect_scores = {}
        
        for aspect, keywords in self.aspect_keywords.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            # Normalize by number of keywords
            confidence = min(matches / max(len(keywords) * 0.3, 1), 1.0)
            aspect_scores[aspect] = confidence
        
        return aspect_scores
    
    def _zero_shot_aspect_detection(self, text: str) -> Dict[str, float]:
        """
        Use zero-shot classification to detect aspects.
        
        Returns confidence score for each aspect label.
        """
        if not self.classifier:
            return self._keyword_aspect_detection(text)
        
        try:
            result = self.classifier(
                text,
                candidate_labels=self.aspect_labels,
                multi_label=True
            )
            
            # Map results to aspect keys
            aspect_scores = {}
            label_to_key = {
                'Product quality and efficacy': 'product_quality',
                'Shipping and delivery logistics': 'shipping_logistics',
                'Packaging and product protection': 'packaging',
                'Customer service and support': 'customer_service',
                'Billing and subscription management': 'billing_subscription'
            }
            
            for label, score in zip(result['labels'], result['scores']):
                key = label_to_key.get(label, label)
                # Only include if confidence > 0.3
                if score > 0.3:
                    aspect_scores[key] = score
            
            return aspect_scores
        except Exception as e:
            logger.warning(f"Zero-shot classification failed: {e}")
            return self._keyword_aspect_detection(text)
    
    def _analyze_sentiment(self, text: str, aspect: str) -> float:
        """
        Analyze sentiment for a specific aspect in the text.
        
        Uses simple rule-based approach with keyword context.
        Score range: -1.0 (very negative) to +1.0 (very positive)
        """
        text_lower = text.lower()
        
        # Positive indicators
        positive_words = [
            'love', 'amazing', 'excellent', 'great', 'fantastic', 'wonderful',
            'perfect', 'best', 'happy', 'pleased', 'satisfied', 'recommend',
            'awesome', 'incredible', 'outstanding', 'superb', 'brilliant',
            'fast', 'quick', 'easy', 'helpful', 'friendly', 'professional'
        ]
        
        # Negative indicators
        negative_words = [
            'hate', 'terrible', 'awful', 'horrible', 'worst', 'bad', 'poor',
            'disappointed', 'frustrated', 'angry', 'upset', 'waste', 'broken',
            'slow', 'difficult', 'rude', 'unprofessional', 'useless', 'never'
        ]
        
        # Negation handling
        negations = ['not', 'no', 'never', "n't", 'neither', 'nowhere']
        
        # Count positive and negative mentions near aspect keywords
        aspect_keywords = self.aspect_keywords.get(aspect, [])
        
        pos_count = 0
        neg_count = 0
        
        # Simple window-based approach
        words = text_lower.split()
        for i, word in enumerate(words):
            # Check if near aspect keyword
            window_start = max(0, i - 5)
            window_end = min(len(words), i + 5)
            window = words[window_start:window_end]
            
            has_aspect = any(kw in window for kw in aspect_keywords)
            if not has_aspect and len(aspect_keywords) > 0:
                continue
            
            # Check for negation before word
            is_negated = False
            for j in range(max(0, i-3), i):
                if words[j] in negations:
                    is_negated = True
                    break
            
            if word in positive_words:
                if is_negated:
                    neg_count += 1
                else:
                    pos_count += 1
            elif word in negative_words:
                if is_negated:
                    pos_count += 1
                else:
                    neg_count += 1
        
        # Calculate sentiment score
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        # Raw score from -1 to 1
        raw_score = (pos_count - neg_count) / total
        
        # Apply sigmoid-like transformation for better distribution
        adjusted_score = np.tanh(raw_score * 1.5)
        
        return round(adjusted_score, 3)
    
    def analyze_review(self, text: str) -> Dict:
        """
        Full aspect-based sentiment analysis for a single review.
        
        Args:
            text: Review text
            
        Returns:
            Dictionary with:
            - aspect_scores: Confidence for each aspect (0-1)
            - sentiment_scores: Sentiment for each aspect (-1 to 1)
            - primary_aspect: Most mentioned aspect
        """
        # Check cache
        cache_key = hash(text)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Detect which aspects are mentioned
        aspect_confidences = self._zero_shot_aspect_detection(text)
        
        # Calculate sentiment for each mentioned aspect
        sentiment_scores = {}
        for aspect, confidence in aspect_confidences.items():
            if confidence > 0.25:  # Threshold for mention
                sentiment = self._analyze_sentiment(text, aspect)
                sentiment_scores[f'sent_{aspect}'] = sentiment
            else:
                sentiment_scores[f'sent_{aspect}'] = 0.0
        
        # Determine primary aspect
        primary_aspect = max(aspect_confidences.items(), key=lambda x: x[1])[0] if aspect_confidences else 'unknown'
        
        result = {
            'aspect_primary': primary_aspect,
            **{f'aspect_{k}': v for k, v in aspect_confidences.items()},
            **sentiment_scores
        }
        
        # Cache result
        self.cache[cache_key] = result
        
        return result
    
    def analyze_dataframe(self, df: pd.DataFrame, 
                          text_column: str = 'cleaned_text',
                          batch_size: int = 100) -> pd.DataFrame:
        """
        Analyze sentiment for entire DataFrame.
        
        Args:
            df: Input DataFrame with review text
            text_column: Name of column containing cleaned text
            batch_size: Number of reviews to process before logging progress
            
        Returns:
            DataFrame with added aspect and sentiment columns
        """
        logger.info(f"Analyzing {len(df)} reviews for aspect-based sentiment...")
        
        df_result = df.copy()
        
        # Initialize columns
        all_aspects = ['product_quality', 'shipping_logistics', 'packaging', 
                       'customer_service', 'billing_subscription']
        
        df_result['aspect_primary'] = 'unknown'
        for aspect in all_aspects:
            df_result[f'aspect_{aspect}'] = 0.0
            df_result[f'sent_{aspect}'] = 0.0
        
        # Process reviews
        for idx, row in df_result.iterrows():
            text = row[text_column]
            
            if not isinstance(text, str) or len(text) < 10:
                continue
            
            result = self.analyze_review(text)
            
            # Update row
            df_result.at[idx, 'aspect_primary'] = result['aspect_primary']
            for aspect in all_aspects:
                df_result.at[idx, f'aspect_{aspect}'] = result.get(f'aspect_{aspect}', 0.0)
                df_result.at[idx, f'sent_{aspect}'] = result.get(f'sent_{aspect}', 0.0)
            
            # Progress logging
            if (idx + 1) % batch_size == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} reviews")
        
        logger.info(f"✅ Sentiment analysis complete")
        
        return df_result


def run_sentiment_analysis(input_path: str = 'data/cleaned_reviews.csv',
                           output_path: str = 'data/enriched_reviews.csv') -> pd.DataFrame:
    """
    Run aspect-based sentiment analysis on cleaned data.
    
    Args:
        input_path: Path to cleaned CSV
        output_path: Path to save enriched CSV
        
    Returns:
        Enriched DataFrame with aspect sentiments
    """
    logger.info(f"Loading cleaned data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Initialize engine
    engine = AspectSentimentEngine()
    
    # Run analysis
    df_enriched = engine.analyze_dataframe(df)
    
    # Save results
    df_enriched.to_csv(output_path, index=False)
    logger.info(f"✅ Saved {len(df_enriched)} enriched reviews to {output_path}")
    
    # Print summary
    print("\n=== Sentiment Analysis Summary ===")
    print(f"Primary aspect distribution:")
    print(df_enriched['aspect_primary'].value_counts())
    print(f"\nAverage sentiment by aspect:")
    for aspect in ['product_quality', 'shipping_logistics', 'packaging', 
                   'customer_service', 'billing_subscription']:
        avg_sent = df_enriched[f'sent_{aspect}'].mean()
        print(f"  {aspect}: {avg_sent:.3f}")
    
    return df_enriched


if __name__ == "__main__":
    df_enriched = run_sentiment_analysis()
    print("\n✅ Phase 3 Complete: Aspect-Based Sentiment Engine finished!")
