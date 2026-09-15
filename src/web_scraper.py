"""
Phase 1: Production-Grade Web Scraper
Scrapes Trustpilot or Amazon reviews with robust error handling.
Includes header rotation, polite delays, and JSON-LD parsing.
"""

import requests
from bs4 import BeautifulSoup
import time
import random
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReviewScraper:
    """
    Production-grade web scraper for e-commerce reviews.
    Features:
    - Header rotation to avoid detection
    - Polite delays between requests
    - JSON-LD structured data parsing (primary)
    - HTML fallback parsing
    - Retry logic with exponential backoff
    - Automatic fallback to synthetic data if blocked
    """
    
    def __init__(self, base_url: str, max_pages: int = 10):
        """
        Initialize scraper.
        
        Args:
            base_url: Starting URL for scraping
            max_pages: Maximum number of pages to scrape
        """
        self.base_url = base_url
        self.max_pages = max_pages
        self.session = requests.Session()
        
        # User agent rotation list
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
        
        # Delay configuration (polite scraping)
        self.min_delay = 2.0
        self.max_delay = 6.0
        
        # Results storage
        self.reviews = []
    
    def _get_headers(self) -> Dict:
        """Get random headers for request."""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        }
    
    def _polite_delay(self):
        """Wait random delay between requests."""
        delay = random.uniform(self.min_delay, self.max_delay)
        logger.info(f"Waiting {delay:.2f}s (polite scraping)...")
        time.sleep(delay)
    
    def _fetch_page(self, url: str, retry_count: int = 3) -> Optional[str]:
        """
        Fetch page content with retry logic.
        
        Args:
            url: URL to fetch
            retry_count: Number of retries on failure
            
        Returns:
            Page HTML content or None if failed
        """
        for attempt in range(retry_count):
            try:
                response = self.session.get(
                    url,
                    headers=self._get_headers(),
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.text
                elif response.status_code == 429:  # Rate limited
                    wait_time = (2 ** attempt) * 5  # Exponential backoff
                    logger.warning(f"Rate limited (429). Waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif response.status_code == 403:  # Forbidden
                    logger.error(f"Access forbidden (403). Site may be blocking scrapers.")
                    return None
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {str(e)}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
        
        return None
    
    def _parse_json_ld(self, soup: BeautifulSoup) -> List[Dict]:
        """
        Parse JSON-LD structured data (preferred method).
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            List of review dictionaries
        """
        reviews = []
        
        # Find JSON-LD script tags
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                
                # Handle different JSON-LD structures
                if isinstance(data, dict):
                    # Single review or aggregate
                    if '@type' in data:
                        if data['@type'] == 'Review':
                            reviews.append(self._extract_review_from_jsonld(data))
                        elif data['@type'] == 'AggregateRating':
                            pass  # Skip aggregate, we want individual reviews
                        elif 'review' in data:
                            # Has nested reviews
                            nested_reviews = data['review']
                            if isinstance(nested_reviews, list):
                                for review in nested_reviews:
                                    reviews.append(self._extract_review_from_jsonld(review))
                            else:
                                reviews.append(self._extract_review_from_jsonld(nested_reviews))
                
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and item.get('@type') == 'Review':
                            reviews.append(self._extract_review_from_jsonld(item))
                            
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                continue
        
        return reviews
    
    def _extract_review_from_jsonld(self, review_data: Dict) -> Dict:
        """Extract review fields from JSON-LD format."""
        return {
            'review_id': f"REV-{hash(str(review_data)) % 100000:05d}",
            'date': review_data.get('datePublished', '')[:10] if review_data.get('datePublished') else '',
            'rating': int(review_data.get('reviewRating', {}).get('ratingValue', 0)),
            'review_text': review_data.get('reviewBody', ''),
            'verified_purchase': True  # Assume verified for scraped data
        }
    
    def _parse_html_fallback(self, soup: BeautifulSoup) -> List[Dict]:
        """
        Fallback HTML parsing when JSON-LD not available.
        Uses multiple selectors to find reviews.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            List of review dictionaries
        """
        reviews = []
        
        # Common review selectors (Trustpilot, Amazon, etc.)
        selectors = [
            {'class': 'review-content'},
            {'class': 'review-body'},
            {'data-hook': 'review-body'},
            {'class': 'rvw-bdy'},
        ]
        
        for selector in selectors:
            elements = soup.find_all(**selector)
            for elem in elements:
                text = elem.get_text(strip=True)
                if len(text) > 20:  # Minimum length filter
                    reviews.append({
                        'review_id': f"REV-{hash(text) % 100000:05d}",
                        'date': '',
                        'rating': 0,  # Would need separate parsing
                        'review_text': text,
                        'verified_purchase': False
                    })
        
        return reviews
    
    def scrape(self) -> pd.DataFrame:
        """
        Main scraping method.
        
        Returns:
            DataFrame of scraped reviews
        """
        logger.info(f"Starting scrape of {self.base_url}")
        logger.info(f"Max pages: {self.max_pages}")
        
        current_url = self.base_url
        page = 1
        
        while page <= self.max_pages and current_url:
            logger.info(f"Scraping page {page}/{self.max_pages}")
            
            html = self._fetch_page(current_url)
            if not html:
                logger.warning("Failed to fetch page. Stopping scrape.")
                break
            
            soup = BeautifulSoup(html, 'lxml')
            
            # Try JSON-LD first, then HTML fallback
            page_reviews = self._parse_json_ld(soup)
            if not page_reviews:
                page_reviews = self._parse_html_fallback(soup)
            
            if page_reviews:
                self.reviews.extend(page_reviews)
                logger.info(f"Found {len(page_reviews)} reviews on page {page}")
            else:
                logger.warning(f"No reviews found on page {page}")
            
            # Check for next page link (implementation varies by site)
            # This is a simplified example
            next_link = soup.find('a', rel='next')
            if next_link and next_link.get('href'):
                current_url = next_link['href']
                if not current_url.startswith('http'):
                    # Relative URL
                    from urllib.parse import urljoin
                    current_url = urljoin(self.base_url, current_url)
            else:
                current_url = None
            
            page += 1
            self._polite_delay()
        
        df = pd.DataFrame(self.reviews)
        logger.info(f"Total reviews scraped: {len(df)}")
        
        return df
    
    def scrape_with_fallback(self) -> pd.DataFrame:
        """
        Scrape with automatic fallback to synthetic data if blocked.
        
        Returns:
            DataFrame of reviews (scraped or synthetic)
        """
        try:
            df = self.scrape()
            
            if len(df) < 10:
                logger.warning("Less than 10 reviews scraped. Using synthetic fallback.")
                return self._use_synthetic_fallback()
            
            return df
            
        except Exception as e:
            logger.error(f"Scraping failed completely: {str(e)}")
            logger.info("Falling back to synthetic data generation...")
            return self._use_synthetic_fallback()
    
    def _use_synthetic_fallback(self) -> pd.DataFrame:
        """Generate synthetic data as fallback."""
        logger.info("Generating synthetic reviews as fallback...")
        from generate_synthetic_data import generate_synthetic_reviews
        return generate_synthetic_reviews(2000)


def scrape_reviews(url: str, max_pages: int = 10) -> pd.DataFrame:
    """
    Convenience function to scrape reviews.
    
    Args:
        url: Starting URL
        max_pages: Maximum pages to scrape
        
    Returns:
        DataFrame of reviews
    """
    scraper = ReviewScraper(url, max_pages)
    return scraper.scrape_with_fallback()


if __name__ == "__main__":
    # Example usage (replace with actual URL)
    # For demo, we'll just generate synthetic data
    print("⚠️  Web scraper configured but no URL provided.")
    print("✅ Use generate_synthetic_data.py for immediate results.")
    print("\nTo scrape a real site:")
    print("  scraper = ReviewScraper('https://example.com/reviews')")
    print("  df = scraper.scrape_with_fallback()")
