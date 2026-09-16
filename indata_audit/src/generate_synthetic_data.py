"""
InData - E-commerce Brand Health & Churn Audit
Module: Synthetic Data Generator
Purpose: Generate 2,000 realistic e-commerce reviews with authentic business complaints.
         This serves as a fallback when web scraping is blocked or rate-limited.
Author: InData Technical Team
"""

import random
import csv
from datetime import datetime, timedelta
from typing import List, Dict
import uuid


class SyntheticReviewGenerator:
    """Generates realistic e-commerce review data for DTC brands."""
    
    def __init__(self, num_reviews: int = 2000):
        self.num_reviews = num_reviews
        self.products = [
            "Radiance Vitamin C Serum",
            "HydraBoost Daily Moisturizer", 
            "PureGlow Cleansing Balm",
            "Youth Renewal Night Cream",
            "Omega-3 Fish Oil Supplement",
            "Probiotic Gut Health Capsules",
            "Collagen Beauty Powder",
            "Turmeric Anti-Inflammatory Tablets"
        ]
        
        # Realistic complaint themes mapped to business aspects
        self.complaint_themes = {
            "Product Quality": [
                "caused breakout within days",
                "made my skin red and irritated",
                "didn't see any results after 3 weeks",
                "pump broke after first use",
                "product separated in the bottle",
                "smells rancid like old oil",
                "consistency is too thick and greasy",
                "caused allergic reaction on my neck",
                "bottle was only half full",
                "texture changed from previous order"
            ],
            "Shipping/Logistics": [
                "shipping took over 3 weeks",
                "package arrived damaged",
                "order never showed up",
                "tracking number didn't work",
                "delivery was delayed by 2 weeks",
                "wrong item was sent",
                "package left in rain",
                "customs held my order for a month",
                "shipping cost more than the product",
                "carrier lost my package twice"
            ],
            "Packaging": [
                "box arrived crushed",
                "bottle leaked all over the box",
                "cap was loose and product spilled",
                "no protective packaging at all",
                "pump dispenser doesn't work",
                "dropper broke inside the bottle",
                "label peeled off immediately",
                "packaging looks cheap for the price",
                "jar arrived cracked",
                "seal was already broken"
            ],
            "Customer Service": [
                "support never responded to my email",
                "waited 45 minutes on chat with no help",
                "representative was rude and unhelpful",
                "got generic copy-paste responses",
                "nobody answered my phone call",
                "they ignored my refund request",
                "agent hung up on me",
                "support said they can't help with this",
                "took 2 weeks to get a response",
                "chat bot kept looping without solution"
            ],
            "Billing/Subscription": [
                "couldn't cancel my subscription",
                "charged twice for same order",
                "hidden fees appeared at checkout",
                "auto-renewal happened without warning",
                "refund still hasn't processed",
                "discount code didn't apply",
                "charged after I cancelled",
                "subscription price increased without notice",
                "hard to find cancel button",
                "they made cancellation intentionally difficult"
            ]
        }
        
        self.positive_phrases = [
            "absolutely love this product",
            "my skin has never looked better",
            "fast shipping and great quality",
            "will definitely repurchase",
            "exceeded my expectations",
            "gentle on sensitive skin",
            "noticed results in just one week",
            "best purchase I've made this year",
            "customer service was super helpful",
            "packaging was beautiful and secure",
            "great value for money",
            "highly recommend to friends",
            "this brand really cares about customers",
            "product works exactly as described",
            "love the natural ingredients"
        ]
        
        self.neutral_phrases = [
            "it's okay nothing special",
            "works but takes time to see results",
            "average product for the price",
            "might work for others but not for me",
            "decent quality but slow shipping",
            "not bad but not amazing either",
            "does what it says but slowly",
            "would consider other options next time"
        ]
        
        self.verified_purchase_weights = [0.85, 0.15]  # 85% verified
        
    def _generate_date(self, start_date: datetime, end_date: datetime) -> str:
        """Generate a random date between start and end."""
        delta = end_date - start_date
        random_days = random.randint(0, delta.days)
        return (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")
    
    def _generate_review_text(self, rating: int, aspect: str = None) -> str:
        """Generate realistic review text based on rating."""
        if rating >= 4:
            phrase = random.choice(self.positive_phrases)
            product_mention = random.choice(self.products)
            templates = [
                f"I {phrase}! The {product_mention} is amazing.",
                f"{phrase.capitalize()}. Been using for 2 months now.",
                f"Just received my order and {phrase}. Great job!",
                f"The {product_mention} exceeded expectations. {phrase.capitalize()}!",
                f"{phrase.capitalize()}! Shipping was fast too."
            ]
            return random.choice(templates)
        
        elif rating == 3:
            phrase = random.choice(self.neutral_phrases)
            return f"{phrase.capitalize()}. It's fine I guess."
        
        else:  # 1-2 stars - detailed complaints
            if aspect is None:
                aspect = random.choice(list(self.complaint_themes.keys()))
            complaint = random.choice(self.complaint_themes[aspect])
            product_mention = random.choice(self.products)
            
            templates = [
                f"Very disappointed. The {complaint}. Won't buy again.",
                f"Terrible experience. {complaint.capitalize()}. Avoid this product.",
                f"Waste of money. {complaint.capitalize()}. Customer service was no help.",
                f"The {product_mention} {complaint}. Very frustrated with this purchase.",
                f"{complaint.capitalize()}. I want a refund immediately.",
                f"DO NOT BUY. {complaint.capitalize()}. This company doesn't care.",
                f"Ordered the {product_mention} and {complaint}. So angry right now.",
                f"One star is too generous. {complaint.capitalize()}."
            ]
            return random.choice(templates)
    
    def generate_reviews(self) -> List[Dict]:
        """Generate the complete dataset of synthetic reviews."""
        reviews = []
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 12, 31)
        
        # Distribution: More negative reviews to simulate churn analysis scenario
        rating_distribution = {
            5: 0.25,
            4: 0.20,
            3: 0.15,
            2: 0.20,
            1: 0.20
        }
        
        for i in range(self.num_reviews):
            # Select rating based on distribution
            rating = random.choices(
                list(rating_distribution.keys()),
                weights=list(rating_distribution.values())
            )[0]
            
            # For low ratings, assign a specific complaint aspect
            aspect = None
            if rating <= 2:
                aspect = random.choice(list(self.complaint_themes.keys()))
            
            review_text = self._generate_review_text(rating, aspect)
            
            review = {
                "review_id": f"REV-{uuid.uuid4().hex[:8].upper()}",
                "date": self._generate_date(start_date, end_date),
                "rating": rating,
                "review_text": review_text,
                "verified_purchase": random.choices([True, False], self.verified_purchase_weights)[0],
                "product_name": random.choice(self.products),
                "customer_name": f"Customer_{random.randint(1000, 9999)}",
                "helpful_count": random.randint(0, 50) if rating != 1 else random.randint(0, 100)
            }
            reviews.append(review)
        
        return reviews
    
    def save_to_csv(self, filename: str = "../data/synthetic_reviews.csv"):
        """Save generated reviews to CSV file."""
        reviews = self.generate_reviews()
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["review_id", "date", "rating", "review_text", 
                         "verified_purchase", "product_name", "customer_name", "helpful_count"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(reviews)
        
        print(f"✓ Successfully generated {len(reviews)} synthetic reviews")
        print(f"✓ Data saved to: {filename}")
        print(f"✓ Columns: {', '.join(fieldnames)}")
        
        # Show sample statistics
        ratings = [r['rating'] for r in reviews]
        print(f"\n📊 Rating Distribution:")
        for rating in sorted(set(ratings)):
            count = ratings.count(rating)
            pct = (count / len(ratings)) * 100
            print(f"   {rating} stars: {count} ({pct:.1f}%)")
        
        return filename


if __name__ == "__main__":
    print("🚀 InData Synthetic Review Generator")
    print("=" * 50)
    generator = SyntheticReviewGenerator(num_reviews=2000)
    generator.save_to_csv()
    print("\n✅ Dataset ready for Phase 2 preprocessing pipeline")
