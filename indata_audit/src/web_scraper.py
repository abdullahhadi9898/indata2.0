"""
InData - E-commerce Brand Health & Churn Audit
Module: Web Scraper for Trustpilot/Amazon Reviews
Purpose: Robust scraper with header rotation, polite delays, and error handling.
         Falls back to synthetic data if scraping is blocked.
Author: InData Technical Team
"""

import requests
from bs4 import BeautifulSoup
import time
import random
import json
import csv
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path


class ReviewScraper:
    """
    Polite web scraper for e-commerce reviews with robust error handling.
    Implements header rotation, random delays, and multiple parsing strategies.
    """
    
    def __init__(self, base_url: str = None, delay_range: tuple = (2, 5)):
        self.base_url = base_url
        self.delay_range = delay_range
        self.session = requests.Session()
        
        # Header rotation pool to avoid detection
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]
        
        self.accept_languages = [
            "en-US,en;q=0.9",
            "en-GB,en;q=0.9",
            "en-CA,en;q=0.9",
            "en-AU,en;q=0.9"
        ]
    
    def _get_random_headers(self) -> Dict[str, str]:
        """Generate random headers to avoid bot detection."""
        return {
            "User-Agent": random.choice(self.user_agents),
            "Accept-Language": random.choice(self.accept_languages),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0"
        }
    
    def _polite_delay(self):
        """Implement random delay between requests."""
        delay = random.uniform(*self.delay_range)
        print(f"⏳ Waiting {delay:.2f} seconds (polite crawling)...")
        time.sleep(delay)
    
    def fetch_page(self, url: str, max_retries: int = 3) -> Optional[str]:
        """
        Fetch a webpage with retry logic and error handling.
        Returns HTML content or None if failed.
        """
        for attempt in range(max_retries):
            try:
                headers = self._get_random_headers()
                response = self.session.get(url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    print(f"✓ Successfully fetched: {url[:50]}...")
                    return response.text
                elif response.status_code == 429:
                    wait_time = 60 * (attempt + 1)
                    print(f"⚠ Rate limited (429). Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                elif response.status_code == 403:
                    print(f"⚠ Access forbidden (403). Rotating headers...")
                    self._polite_delay()
                else:
                    print(f"⚠ HTTP Error {response.status_code}. Retrying...")
                    self._polite_delay()
                    
            except requests.exceptions.RequestException as e:
                print(f"⚠ Request error (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    self._polite_delay()
                else:
                    print(f"❌ Failed to fetch after {max_retries} attempts")
                    return None
        
        return None
    
    def parse_json_ld(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract review data from JSON-LD structured data."""
        reviews = []
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                
                # Handle different JSON-LD structures
                if isinstance(data, dict):
                    if 'review' in data:
                        review_data = data['review']
                        if isinstance(review_data, list):
                            reviews.extend(review_data)
                        else:
                            reviews.append(review_data)
                    elif '@type' in data and data['@type'] == 'Review':
                        reviews.append(data)
                
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and item.get('@type') == 'Review':
                            reviews.append(item)
                            
            except (json.JSONDecodeError, AttributeError) as e:
                continue
        
        return reviews
    
    def parse_html_reviews(self, soup: BeautifulSoup) -> List[Dict]:
        """Fallback: Extract reviews from HTML structure."""
        reviews = []
        
        # Common review selectors (customize per target site)
        review_selectors = [
            {'container': '[data-hook="review"]', 'text': '.review-text', 'rating': '.review-rating'},
            {'container': '.review-item', 'text': '.review-content', 'rating': '.star-rating'},
            {'container': '.trustpilot-review', 'text': '.review-text', 'rating': '.star-rating'},
            {'container': '[itemtype="http://schema.org/Review"]', 'text': '[itemprop="reviewBody"]', 'rating': '[itemprop="reviewRating"]'}
        ]
        
        for selector in review_selectors:
            containers = soup.select(selector['container'])
            if containers:
                for container in containers:
                    try:
                        text_elem = container.select_one(selector['text'])
                        rating_elem = container.select_one(selector['rating'])
                        
                        if text_elem:
                            review_text = text_elem.get_text(strip=True)
                            rating = self._extract_rating(rating_elem) if rating_elem else None
                            
                            if review_text and len(review_text) > 10:
                                reviews.append({
                                    'review_text': review_text,
                                    'rating': rating,
                                    'date': None,
                                    'verified_purchase': False
                                })
                    except Exception:
                        continue
                break
        
        return reviews
    
    def _extract_rating(self, rating_elem) -> Optional[int]:
        """Extract numeric rating from various formats."""
        try:
            # Try to get from attribute
            if 'content' in rating_elem.attrs:
                return int(float(rating_elem['content']))
            
            # Try to get from text
            text = rating_elem.get_text(strip=True)
            if '/' in text:
                return int(text.split('/')[0])
            
            # Try to count stars
            stars = rating_elem.find_all('span', class_='star')
            if stars:
                return len([s for s in stars if 'filled' in s.get('class', [])])
                
        except (ValueError, AttributeError):
            pass
        
        return None
    
    def normalize_review(self, raw_review: Dict, source: str = "scraped") -> Dict:
        """Normalize scraped review to standard format."""
        import uuid
        from datetime import datetime
        
        review_text = raw_review.get('reviewBody', raw_review.get('review_text', ''))
        rating = raw_review.get('reviewRating', raw_review.get('rating'))
        
        # Handle nested rating objects
        if isinstance(rating, dict):
            rating = rating.get('ratingValue')
        
        if isinstance(rating, str):
            try:
                rating = int(float(rating))
            except ValueError:
                rating = None
        
        date_str = raw_review.get('datePublished', raw_review.get('date', ''))
        if date_str:
            try:
                date_str = date_str[:10]  # Extract YYYY-MM-DD
            except:
                date_str = datetime.now().strftime("%Y-%m-%d")
        else:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        return {
            'review_id': f"SCR-{uuid.uuid4().hex[:8].upper()}",
            'date': date_str,
            'rating': rating,
            'review_text': review_text.strip() if review_text else '',
            'verified_purchase': raw_review.get('verifiedPurchase', False),
            'product_name': raw_review.get('itemReviewed', {}).get('name', 'Unknown'),
            'customer_name': raw_review.get('author', {}).get('name', 'Anonymous'),
            'source': source
        }
    
    def scrape_reviews(self, urls: List[str], output_file: str = "../data/scraped_reviews.csv") -> bool:
        """
        Main scraping method. Iterates through URLs and saves results.
        Falls back to synthetic data if all scraping attempts fail.
        """
        all_reviews = []
        
        for i, url in enumerate(urls, 1):
            print(f"\n📄 Scraping page {i}/{len(urls)}: {url[:50]}...")
            
            html = self.fetch_page(url)
            if not html:
                continue
            
            soup = BeautifulSoup(html, 'lxml')
            
            # Try JSON-LD first (most reliable)
            raw_reviews = self.parse_json_ld(soup)
            
            # Fallback to HTML parsing
            if not raw_reviews:
                print("ℹ No JSON-LD found, trying HTML parsing...")
                raw_reviews = self.parse_html_reviews(soup)
            
            if raw_reviews:
                print(f"✓ Found {len(raw_reviews)} reviews on this page")
                for raw in raw_reviews:
                    normalized = self.normalize_review(raw, source=url)
                    if normalized['review_text']:
                        all_reviews.append(normalized)
            else:
                print(f"⚠ No reviews found on this page")
            
            if i < len(urls):
                self._polite_delay()
        
        # Save results
        if all_reviews:
            self._save_to_csv(all_reviews, output_file)
            print(f"\n✅ Successfully scraped {len(all_reviews)} reviews")
            return True
        else:
            print("\n❌ Scraping failed. Falling back to synthetic data generator...")
            return False
    
    def _save_to_csv(self, reviews: List[Dict], filename: str):
        """Save scraped reviews to CSV."""
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        fieldnames = ["review_id", "date", "rating", "review_text", 
                     "verified_purchase", "product_name", "customer_name", "source"]
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(reviews)
        
        print(f"✓ Data saved to: {filename}")


def main():
    """Example usage of the scraper."""
    print("🚀 InData Review Scraper")
    print("=" * 50)
    
    scraper = ReviewScraper(delay_range=(3, 6))
    
    # Example URLs (replace with actual target URLs)
    # Note: These are placeholders - actual scraping requires permission
    sample_urls = [
        "https://www.trustpilot.com/review/example-store.com",
        # Add more URLs as needed
    ]
    
    success = scraper.scrape_reviews(sample_urls)
    
    if not success:
        print("\n💡 Switching to synthetic data fallback...")
        print("Run: python generate_synthetic_data.py")


if __name__ == "__main__":
    main()
