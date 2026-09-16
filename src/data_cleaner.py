"""
Phase 2: Data Cleaning & Preprocessing Pipeline
Modular cleaning class for review data preprocessing.
Handles missing values, HTML removal, spam detection, and NLP preparation.
"""

import pandas as pd
import numpy as np
import re
import html
from typing import List, Optional
import logging
import spacy
from nltk.corpus import stopwords

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("spaCy model 'en_core_web_sm' not found.")
    logger.error("Run: python -m spacy download en_core_web_sm")
    raise


class ReviewCleaner:
    """
    Modular review data cleaning engine.
    
    Features:
    - Text normalization (HTML removal, URL stripping, whitespace)
    - Spam/bot detection and filtering
    - Smart lemmatization with negation preservation
    - Encoding error correction
    """
    
    def __init__(self):
        """Initialize cleaner with NLP models and stop words."""
        self.nlp = nlp
        
        # Extended stop words (keep negations!)
        base_stopwords = set(stopwords.words('english'))
        
        # Remove negations from stop words - CRITICAL for sentiment
        negations_to_keep = {'not', 'no', 'nor', 'never', "n't", 'neither', 'nowhere'}
        self.stop_words = base_stopwords - negations_to_keep
        
        # Add domain-specific noise words
        domain_noise = {
            'bought', 'received', 'got', 'ordered', 'purchased', 'product',
            'item', 'thing', 'stuff', 'use', 'using', 'used', 'one', 'really',
            'would', 'could', 'should', 'also', 'even', 'just', 'much', 'many'
        }
        self.stop_words.update(domain_noise)
        
        # Bot/spam patterns
        self.bot_patterns = [
            r'great deal', r'fast shipping', r'highly recommend',
            r'click here', r'buy now', r'amazing product',
            r'five stars', r'best seller', r'cheap price',
            r'visit website', r'free shipping', r'awesome quality'
        ]
        
        # Compile regex patterns for efficiency
        self.html_pattern = re.compile(r'<[^>]+>')
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.whitespace_pattern = re.compile(r'\s+')
        self.punctuation_pattern = re.compile(r'[^\w\s\'-]')
        
    def remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text."""
        if not isinstance(text, str):
            return ""
        text = html.unescape(text)  # Handle entities like &amp;
        return self.html_pattern.sub('', text)
    
    def remove_urls_and_emails(self, text: str) -> str:
        """Remove URLs and email addresses."""
        if not isinstance(text, str):
            return ""
        text = self.url_pattern.sub('', text)
        text = self.email_pattern.sub('', text)
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace to single spaces."""
        if not isinstance(text, str):
            return ""
        return self.whitespace_pattern.sub(' ', text).strip()
    
    def fix_encoding_errors(self, text: str) -> str:
        """Fix common encoding errors."""
        if not isinstance(text, str):
            return ""
        
        # Common encoding fixes
        replacements = {
            'â€™': "'", 'â€˜': "'", 'â€œ': '"', 'â€': '"',
            'Ã©': 'é', 'Ã¨': 'è', 'Ã¼': 'ü', 'Ã±': 'ñ',
            '...': '…', '--': '—'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def remove_excessive_punctuation(self, text: str) -> str:
        """Remove excessive punctuation (!!!, ???, etc.)."""
        if not isinstance(text, str):
            return ""
        
        # Replace multiple punctuation with single
        text = re.sub(r'!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)
        text = re.sub(r'\.{3,}', '...', text)
        
        return text
    
    def is_bot_review(self, text: str) -> bool:
        """
        Detect bot/spam reviews.
        
        Criteria:
        - Length < 15 characters
        - Contains repetitive bot phrases
        - Excessive capitalization (>50% caps)
        """
        if not isinstance(text, str) or len(text) < 15:
            return True
        
        text_lower = text.lower()
        
        # Check for bot phrases (2 or more matches = likely bot)
        bot_matches = sum(1 for pattern in self.bot_patterns if pattern in text_lower)
        if bot_matches >= 2:
            return True
        
        # Check for excessive caps
        alpha_chars = [c for c in text if c.isalpha()]
        if len(alpha_chars) > 0:
            caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if caps_ratio > 0.5:
                return True
        
        return False
    
    def lemmatize_with_negations(self, text: str) -> str:
        """
        Lemmatize text while preserving negations.
        
        This is CRITICAL for accurate sentiment analysis.
        Words like "not", "never" must be kept.
        """
        if not isinstance(text, str) or len(text) < 5:
            return text
        
        doc = self.nlp(text.lower())
        lemmatized = []
        
        for token in doc:
            # Keep negations and important modifiers
            if token.text in {'not', 'no', 'nor', 'never', "n't", 'neither', 'nowhere'}:
                lemmatized.append(token.text)
            elif token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV'} and token.text not in self.stop_words:
                lemmatized.append(token.lemma_)
            elif token.is_alpha and len(token.text) > 2:
                # Keep some other meaningful words
                lemmatized.append(token.lemma_)
        
        return ' '.join(lemmatized)
    
    def clean_text(self, text: str, full_clean: bool = True) -> str:
        """
        Apply full cleaning pipeline to text.
        
        Args:
            text: Raw review text
            full_clean: If True, apply all transformations including lemmatization
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Step 1: Fix encoding
        text = self.fix_encoding_errors(text)
        
        # Step 2: Remove HTML
        text = self.remove_html_tags(text)
        
        # Step 3: Remove URLs and emails
        text = self.remove_urls_and_emails(text)
        
        # Step 4: Normalize whitespace
        text = self.normalize_whitespace(text)
        
        # Step 5: Remove excessive punctuation
        text = self.remove_excessive_punctuation(text)
        
        # Step 6: Full NLP cleaning (optional)
        if full_clean:
            text = self.lemmatize_with_negations(text)
        
        return text.strip()
    
    def clean_dataframe(self, df: pd.DataFrame, text_column: str = 'review_text') -> pd.DataFrame:
        """
        Clean entire DataFrame of reviews.
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing review text
            
        Returns:
            DataFrame with cleaned text and spam flags
        """
        logger.info(f"Cleaning {len(df)} reviews...")
        
        # Create copy to avoid modifying original
        df_clean = df.copy()
        
        # Check for bot/spam
        logger.info("Detecting bot/spam reviews...")
        df_clean['is_bot'] = df_clean[text_column].apply(self.is_bot_review)
        
        # Count bots before removal
        bot_count = df_clean['is_bot'].sum()
        logger.info(f"Found {bot_count} potential bot/spam reviews ({bot_count/len(df)*100:.1f}%)")
        
        # Remove bot reviews
        df_clean = df_clean[~df_clean['is_bot']].copy()
        
        # Clean text
        logger.info("Cleaning review text...")
        df_clean['cleaned_text'] = df_clean[text_column].apply(lambda x: self.clean_text(x, full_clean=True))
        
        # Remove empty reviews after cleaning
        df_clean = df_clean[df_clean['cleaned_text'].str.len() > 10].copy()
        
        # Log results
        logger.info(f"Reviews after cleaning: {len(df_clean)} (removed {len(df) - len(df_clean)})")
        
        return df_clean
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in review DataFrame.
        
        Strategy:
        - Drop rows with missing review_text
        - Fill missing ratings with median
        - Fill missing dates with earliest date
        - Fill missing verified_purchase with False
        """
        df_clean = df.copy()
        
        # Drop rows with missing review text
        initial_count = len(df_clean)
        df_clean = df_clean.dropna(subset=['review_text'])
        dropped = initial_count - len(df_clean)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with missing review text")
        
        # Fill missing ratings with median
        if 'rating' in df_clean.columns:
            median_rating = df_clean['rating'].median()
            df_clean['rating'] = df_clean['rating'].fillna(median_rating)
        
        # Fill missing dates
        if 'date' in df_clean.columns:
            earliest_date = df_clean['date'].min()
            df_clean['date'] = df_clean['date'].fillna(earliest_date)
        
        # Fill missing verified_purchase
        if 'verified_purchase' in df_clean.columns:
            df_clean['verified_purchase'] = df_clean['verified_purchase'].fillna(False)
        
        return df_clean


def load_and_clean_data(input_path: str = 'data/synthetic_reviews.csv',
                        output_path: str = 'data/cleaned_reviews.csv') -> pd.DataFrame:
    """
    Load raw data, clean it, and save results.
    
    Args:
        input_path: Path to raw CSV
        output_path: Path to save cleaned CSV
        
    Returns:
        Cleaned DataFrame
    """
    logger.info(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Initialize cleaner
    cleaner = ReviewCleaner()
    
    # Handle missing values first
    df = cleaner.handle_missing_values(df)
    
    # Clean text and remove spam
    df_clean = cleaner.clean_dataframe(df, text_column='review_text')
    
    # Save results
    df_clean.to_csv(output_path, index=False)
    logger.info(f"✅ Saved {len(df_clean)} cleaned reviews to {output_path}")
    
    return df_clean


if __name__ == "__main__":
    # Run cleaning pipeline
    df_clean = load_and_clean_data()
    
    print("\n=== Cleaning Summary ===")
    print(f"Final review count: {len(df_clean)}")
    print(f"\nSample cleaned reviews:")
    print(df_clean[['review_id', 'rating', 'cleaned_text']].head(3).to_string())
    print("\n✅ Phase 2 Complete: Data cleaning finished!")
