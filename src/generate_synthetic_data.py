"""
Phase 1: Synthetic Data Generator
Generates realistic e-commerce review data for churn analysis.
Creates 2,000 reviews with authentic business complaints across 5 core aspects.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import uuid

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Business aspect templates with realistic complaints
ASPECT_TEMPLATES = {
    'product_quality': {
        'positive': [
            "This product changed my skin completely. Absolutely love it!",
            "Best serum I've ever used. My face feels amazing.",
            "The texture is perfect and it absorbs quickly.",
            "Finally found a moisturizer that doesn't break me out.",
            "Results visible in just one week. Highly recommend!",
            "The scent is subtle and pleasant. Works wonders.",
            "My acne has cleared up significantly since using this.",
            "Worth every penny. My skin has never looked better.",
            "Gentle on sensitive skin but still effective.",
            "The ingredients list is clean and transparent."
        ],
        'negative': [
            "This caused a massive breakout on my chin.",
            "The pump broke after two weeks of use.",
            "Product separated and smelled rancid.",
            "Made my skin extremely dry and flaky.",
            "Nothing happened. Complete waste of money.",
            "The formula feels greasy and sits on top of skin.",
            "Caused an allergic reaction - had to see a dermatologist.",
            "Inconsistent texture between batches.",
            "The active ingredients seem diluted compared to before.",
            "Left a weird film on my face that wouldn't wash off."
        ]
    },
    'shipping_logistics': {
        'positive': [
            "Arrived two days early! Impressed with the speed.",
            "Tracking was accurate and delivery was smooth.",
            "Package came in perfect condition, well protected.",
            "Fast shipping even during the holiday rush.",
            "Carrier was professional and careful with delivery."
        ],
        'negative': [
            "Took 3 weeks to arrive when promised 5 days.",
            "Package was lost and customer service didn't help.",
            "Shipping took forever. Still waiting after a month.",
            "Carrier left package in the rain - completely soaked.",
            "No tracking updates for 2 weeks. Very frustrating.",
            "Received someone else's order. Major mix-up.",
            "Had to pick up from depot because they couldn't deliver.",
            "Customs held it for weeks due to improper paperwork.",
            "Delivery driver threw package at my door.",
            "Shipping cost was ridiculous for such a small item."
        ]
    },
    'packaging': {
        'positive': [
            "Love the eco-friendly packaging! So thoughtful.",
            "Jar arrived perfectly sealed with no leaks.",
            "Beautiful unboxing experience. Felt luxurious.",
            "Minimalist design and recyclable materials.",
            "Sturdy box protected the bottle perfectly."
        ],
        'negative': [
            "Bottle leaked everywhere inside the box.",
            "Jar arrived cracked - product was contaminated.",
            "Box was crushed and the bottle was broken.",
            "Too much plastic packaging for such a small product.",
            "Cap was loose and product spilled during shipping.",
            "Packaging looks cheap and flimsy.",
            "No seal on the bottle - concerned about tampering.",
            "The dropper was broken inside the bottle.",
            "Label peeled off after first use.",
            "Impossible to get the last bit of product out."
        ]
    },
    'customer_service': {
        'positive': [
            "Support team resolved my issue within hours.",
            "Very helpful and knowledgeable staff.",
            "They went above and beyond to help me.",
            "Quick response time and friendly tone.",
            "Excellent follow-up after my initial complaint."
        ],
        'negative': [
            "Waited 45 minutes on hold with no answer.",
            "Support never responded to my email.",
            "Agent was rude and dismissive of my concern.",
            "Got transferred 5 times and still no resolution.",
            "They refused to honor their return policy.",
            "Chat bot was useless and couldn't connect to human.",
            "No phone support available - only email.",
            "Response was copy-pasted and didn't address my issue.",
            "They charged me for return shipping despite their error.",
            "Account was closed without warning or explanation."
        ]
    },
    'billing_subscription': {
        'positive': [
            "Subscription is easy to manage and pause.",
            "Transparent pricing with no hidden fees.",
            "Refund was processed quickly and fairly.",
            "Love the auto-ship discount program.",
            "Cancellation was straightforward with no hassle."
        ],
        'negative': [
            "Couldn't cancel subscription - kept getting charged.",
            "Was charged twice for the same order.",
            "Hidden fees appeared at checkout.",
            "They made cancellation intentionally difficult.",
            "Subscription renewed without my consent.",
            "Refund took 6 weeks to process.",
            "Price increased without any notification.",
            "Was charged for items I explicitly removed from cart.",
            "Credit card was charged before item shipped.",
            "Loyalty points disappeared from my account."
        ]
    }
}

# Neutral/filler phrases to add variety
FILLER_PHRASES = [
    "Overall decent experience.",
    "Would consider buying again.",
    "Not sure if I'd recommend to friends.",
    "Mixed feelings about this purchase.",
    "It's okay but nothing special.",
    "Expected more based on the reviews.",
    "Might work better for others.",
    "Average product for the price point.",
    "Have seen better alternatives.",
    "Does what it claims but barely."
]


def generate_review_id():
    """Generate unique review ID."""
    return f"REV-{uuid.uuid4().hex[:8].upper()}"


def generate_date(start_date, end_date):
    """Generate random date between start and end."""
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    return start_date + timedelta(days=random_days)


def generate_rating(aspect_sentiments):
    """Generate rating based on aspect sentiments."""
    # Calculate average sentiment
    avg_sentiment = np.mean([v for v in aspect_sentiments.values() if v != 0])
    
    # Add some noise
    noise = random.uniform(-0.3, 0.3)
    adjusted = avg_sentiment + noise
    
    # Map to 1-5 scale
    if adjusted > 0.5:
        return random.choices([4, 5], weights=[0.3, 0.7])[0]
    elif adjusted > 0:
        return random.choices([3, 4], weights=[0.4, 0.6])[0]
    elif adjusted > -0.3:
        return random.choices([2, 3], weights=[0.5, 0.5])[0]
    else:
        return random.choices([1, 2], weights=[0.6, 0.4])[0]


def generate_review_text(aspect_sentiments, include_filler=False):
    """Generate realistic review text based on sentiments."""
    sentences = []
    
    for aspect, sentiment in aspect_sentiments.items():
        if sentiment == 0:
            continue
            
        if sentiment > 0:
            sentences.append(random.choice(ASPECT_TEMPLATES[aspect]['positive']))
        else:
            sentences.append(random.choice(ASPECT_TEMPLATES[aspect]['negative']))
    
    # Add filler for neutral reviews
    if include_filler and len(sentences) == 0:
        sentences.append(random.choice(FILLER_PHRASES))
    
    # Shuffle and join
    random.shuffle(sentences)
    return " ".join(sentences)


def generate_synthetic_reviews(n_reviews=2000):
    """
    Generate synthetic e-commerce reviews with realistic business complaints.
    
    Args:
        n_reviews: Number of reviews to generate (default: 2000)
    
    Returns:
        pandas DataFrame with columns:
        - review_id
        - date
        - rating
        - review_text
        - verified_purchase
    """
    print(f"Generating {n_reviews} synthetic reviews...")
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    reviews = []
    aspects = list(ASPECT_TEMPLATES.keys())
    
    # Create realistic distribution of issues
    # More negative reviews for churn analysis scenario
    for i in range(n_reviews):
        review_id = generate_review_id()
        date = generate_date(start_date, end_date)
        
        # Determine which aspects have issues (realistic correlation)
        # Product quality issues often correlate with negative overall rating
        aspect_sentiments = {aspect: 0 for aspect in aspects}
        
        # Randomly assign 1-3 aspects per review
        n_aspects_mentioned = random.randint(1, 3)
        mentioned_aspects = random.sample(aspects, n_aspects_mentioned)
        
        # Assign sentiments with bias toward negative (churn scenario)
        negative_bias = 0.45  # 45% chance of negative sentiment
        positive_bias = 0.40  # 40% chance of positive sentiment
        neutral_bias = 0.15   # 15% chance aspect mentioned neutrally
        
        for aspect in mentioned_aspects:
            rand_val = random.random()
            if rand_val < negative_bias:
                aspect_sentiments[aspect] = random.uniform(-1.0, -0.3)
            elif rand_val < negative_bias + positive_bias:
                aspect_sentiments[aspect] = random.uniform(0.3, 1.0)
            else:
                aspect_sentiments[aspect] = random.uniform(-0.2, 0.2)
        
        # Generate rating based on sentiments
        rating = generate_rating(aspect_sentiments)
        
        # Generate review text
        include_filler = (rating == 3)
        review_text = generate_review_text(aspect_sentiments, include_filler)
        
        # Verified purchase (85% verified)
        verified = random.random() < 0.85
        
        reviews.append({
            'review_id': review_id,
            'date': date.strftime('%Y-%m-%d'),
            'rating': rating,
            'review_text': review_text,
            'verified_purchase': verified
        })
    
    df = pd.DataFrame(reviews)
    
    # Print summary statistics
    print("\n=== Synthetic Data Summary ===")
    print(f"Total Reviews: {len(df)}")
    print(f"\nRating Distribution:")
    print(df['rating'].value_counts().sort_index())
    print(f"\nVerified Purchase Rate: {df['verified_purchase'].mean():.1%}")
    print(f"\nDate Range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nSample Reviews:")
    print(df.head(3)[['review_id', 'rating', 'review_text']].to_string())
    
    return df


def save_to_csv(df, filepath='data/synthetic_reviews.csv'):
    """Save generated reviews to CSV."""
    df.to_csv(filepath, index=False)
    print(f"\n✅ Saved {len(df)} reviews to {filepath}")


if __name__ == "__main__":
    # Generate and save synthetic data
    df = generate_synthetic_reviews(2000)
    save_to_csv(df)
    print("\n✅ Phase 1 Complete: Synthetic data generation finished!")
