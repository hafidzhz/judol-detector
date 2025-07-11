# scraper.py
import requests
import pandas as pd
import time
import os
import json
import random
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("youtube_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("youtube_scraper")

# Constants
API_KEYS = [
    # Add more API keys here for dynamic changes when the limit is off
]
REGION_CODE = 'ID'
MAX_RESULTS = 100
OUTPUT_FILE = 'raw_youtube_comments_{date}.csv'.format(date=datetime.now().strftime('%d_%m_%Y'))
DELIMITER = '¤'
CACHE_DIR = 'cache'
QUOTA_LIMIT = 7000  # Daily quota limit per API key

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Track API usage
api_usage = {key: 0 for key in API_KEYS}
current_api_key_index = 0

def get_api_key():
    """Get the next API key with available quota"""
    global current_api_key_index
    
    # Try to find a key with remaining quota
    start_index = current_api_key_index
    while True:
        key = API_KEYS[current_api_key_index]
        if api_usage[key] < QUOTA_LIMIT:
            return key
        
        # Move to next key
        current_api_key_index = (current_api_key_index + 1) % len(API_KEYS)
        
        # If we've checked all keys and come back to the start, raise exception
        if current_api_key_index == start_index:
            logger.error("All API keys have reached their quota limits")
            raise Exception("API quota exceeded for all keys")
    
    return API_KEYS[current_api_key_index]

def update_api_usage(api_key, units=1):
    """Update the API usage counter"""
    api_usage[api_key] += units
    logger.debug(f"API key {api_key[-8:]} usage: {api_usage[api_key]}/{QUOTA_LIMIT}")

def load_cache(cache_file):
    """Load data from cache file if it exists"""
    cache_path = os.path.join(CACHE_DIR, cache_file)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_file}: {e}")
    return None

def save_cache(cache_file, data):
    """Save data to cache file"""
    cache_path = os.path.join(CACHE_DIR, cache_file)
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to save cache {cache_file}: {e}")

def is_cache_valid(cache_file, max_age_hours=24):
    """Check if cache file is still valid based on age"""
    cache_path = os.path.join(CACHE_DIR, cache_file)
    if not os.path.exists(cache_path):
        return False
    
    file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
    max_age = timedelta(hours=max_age_hours)
    
    return datetime.now() - file_mod_time < max_age

def fetch_indonesian_popular_by_categories(region_code='ID', use_cache=True):
    """Fetch popular Indonesian videos by categories with caching"""
    cache_file = f"popular_videos_{region_code}.json"
    
    # Try to use cache if it's valid
    if use_cache and is_cache_valid(cache_file, 12):  # Cache valid for 12 hours
        cached_data = load_cache(cache_file)
        if cached_data:
            logger.info(f"Using cached video IDs from {cache_file}")
            return cached_data

    # Define category queries more efficiently
    category_queries = {
        '1': ['film indonesia', 'animasi'],  # Film & Animation
        '10': ['musik indonesia', 'lagu'],  # Music
        '17': ['olahraga indonesia', 'sepak bola'],  # Sports
        '20': ['gameplay indonesia', 'game'],  # Gaming
        '22': ['vlog indonesia'],  # People & Blogs
        '23': ['komedi indonesia'],  # Comedy
        '24': ['infotainment indonesia'],  # Entertainment
        '25': ['berita indonesia'],  # News & Politics
    }

    all_video_ids = set()
    max_per_category = 50  # Reduced from 200 to save quota
    page_size = 50
    
    for category_id, keywords in category_queries.items():
        # Only use one keyword per category to save quota
        query = random.choice(keywords)
        fetched = 0
        url = 'https://www.googleapis.com/youtube/v3/search'
        
        api_key = get_api_key()
        params = {
            'part': 'snippet',
            'type': 'video',
            'videoCategoryId': category_id,
            'regionCode': region_code,
            'relevanceLanguage': 'id',
            'maxResults': page_size,
            'q': query,
            'order': 'viewCount',
            'key': api_key
        }

        logger.info(f'🔍 Category {category_id} | Query: "{query}"')
        
        # One request per category (costs 100 units)
        try:
            response = requests.get(url, params=params).json()
            update_api_usage(api_key, 100)  # Search costs 100 units
            
            video_ids = [item['id']['videoId'] for item in response.get('items', []) if 'videoId' in item['id']]
            all_video_ids.update(video_ids)
            
            logger.info(f"Found {len(video_ids)} videos for category {category_id}")
        except Exception as e:
            logger.error(f"Error fetching category {category_id}: {e}")

    # Save to cache
    video_ids_list = list(all_video_ids)
    save_cache(cache_file, video_ids_list)
    
    logger.info(f'✅ Total unique Indonesian videos: {len(video_ids_list)}')
    return video_ids_list

def fetch_trending_video_ids(region_code='ID', max_results=50, use_cache=True):
    """Fetch trending videos with caching"""
    cache_file = f"trending_videos_{region_code}.json"
    
    # Try to use cache if it's valid
    if use_cache and is_cache_valid(cache_file, 4):  # Cache valid for 4 hours
        cached_data = load_cache(cache_file)
        if cached_data:
            logger.info(f"Using cached trending video IDs from {cache_file}")
            return cached_data

    url = 'https://www.googleapis.com/youtube/v3/videos'
    video_ids = []
    
    api_key = get_api_key()
    params = {
        'part': 'snippet',
        'chart': 'mostPopular',
        'regionCode': region_code,
        'maxResults': max_results,
        'key': api_key
    }

    try:
        response = requests.get(url, params=params)
        update_api_usage(api_key, 1)  # Videos list costs 1 unit
        
        data = response.json()
        video_ids = [item['id'] for item in data.get('items', [])]
        
        # Save to cache
        save_cache(cache_file, video_ids)
        
        logger.info(f"Fetched {len(video_ids)} trending videos")
    except Exception as e:
        logger.error(f"Error fetching trending videos: {e}")
    
    return video_ids

def check_comments_enabled(video_id):
    """Check if comments are enabled for a video using cached video data"""
    cache_file = f"video_info_{video_id}.json"
    
    # Try to use cache if it exists
    cached_data = load_cache(cache_file)
    if cached_data:
        return cached_data.get('comments_enabled', True)
    
    # We'll assume comments are enabled and let the comment fetch handle errors
    return True

def fetch_comments(video_id, max_comments=100):
    """Fetch comments for a video with caching"""
    cache_file = f"comments_{video_id}.json"
    
    # Try to use cache if it exists
    cached_data = load_cache(cache_file)
    if cached_data:
        logger.info(f"Using cached comments for video {video_id}")
        return cached_data
    
    comments = []
    url = 'https://www.googleapis.com/youtube/v3/commentThreads'
    
    api_key = get_api_key()
    params = {
        'part': 'snippet',
        'videoId': video_id,
        'key': api_key,
        'textFormat': 'plainText'
    }

    try:
        response = requests.get(url, params=params)
        update_api_usage(api_key, 1)  # CommentThreads list costs 1 unit
        
        data = response.json()
        
        # Check for errors
        if 'error' in data:
            error_reason = data['error']['errors'][0]['reason']
            if error_reason == 'commentsDisabled':
                # Save to cache that comments are disabled
                save_cache(f"video_info_{video_id}.json", {'comments_enabled': False})
                logger.info(f"Comments are disabled for video {video_id}")
                return []
            elif error_reason == 'quotaExceeded':
                logger.error(f"Quota exceeded for API key {api_key[-8:]}")
                raise Exception("API quota exceeded")
            else:
                logger.warning(f"API error for video {video_id}: {error_reason}")
                return []
        
        # Process comments
        for item in data.get('items', []):
            top_comment = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'videoId': video_id,
                'username': top_comment['authorDisplayName'],
                'comment': top_comment['textOriginal'],
                'likeCount': top_comment.get('likeCount', 0),
                'publishedAt': top_comment['publishedAt']
            })

        # We'll only fetch one page to save quota (not using nextPageToken)
        
        # Save to cache
        save_cache(cache_file, comments)
        
        logger.info(f"Fetched {len(comments)} comments for video {video_id}")
    except Exception as e:
        logger.error(f"Error fetching comments for video {video_id}: {e}")
    
    return comments

def get_combined_video_ids(region_code='ID', max_results=50, use_cache=True):
    """Get combined video IDs from trending and search with caching"""
    cache_file = "combined_video_ids.json"
    
    # Try to use cache if it's valid
    if use_cache and is_cache_valid(cache_file, 6):  # Cache valid for 6 hours
        cached_data = load_cache(cache_file)
        if cached_data:
            logger.info(f"Using cached combined video IDs from {cache_file}")
            return cached_data
    
    # Get trending videos (lower quota impact)
    trending_ids = fetch_trending_video_ids(region_code, max_results, use_cache)
    
    # Get search-based videos (higher quota impact)
    search_ids = fetch_indonesian_popular_by_categories(region_code, use_cache)
    
    # Combine and remove duplicates
    combined_ids = list(set(trending_ids + search_ids))
    random.shuffle(combined_ids)  # Randomize to get diverse content
    
    # Save to cache
    save_cache(cache_file, combined_ids)
    
    logger.info(f'🧩 Combined {len(combined_ids)} unique video IDs.')
    return combined_ids

def clean_text(text):
    """Clean text for CSV output"""
    if not isinstance(text, str):
        text = str(text)
    return text.replace('\n', ' ').replace('\r', ' ').replace(DELIMITER, ' ').strip()

def load_processed_videos():
    """Load list of already processed videos"""
    processed_file = os.path.join(CACHE_DIR, "processed_videos.json")
    if os.path.exists(processed_file):
        try:
            with open(processed_file, 'r') as f:
                return set(json.load(f))
        except:
            pass
    return set()

def save_processed_video(video_id):
    """Save video ID to processed list"""
    processed_file = os.path.join(CACHE_DIR, "processed_videos.json")
    processed = load_processed_videos()
    processed.add(video_id)
    
    try:
        with open(processed_file, 'w') as f:
            json.dump(list(processed), f)
    except Exception as e:
        logger.warning(f"Failed to save processed video list: {e}")

def scrape_comments(max_videos=None, max_comments_per_video=50):
    """Main function to scrape comments with quota optimization"""
    # Load already processed videos to avoid duplicates
    processed_videos = load_processed_videos()
    
    # Get video IDs
    all_video_ids = get_combined_video_ids(REGION_CODE, 50)
    logger.info(f'🎥 Found {len(all_video_ids)} videos to process.')
    
    # Remove already processed videos
    video_ids = [vid for vid in all_video_ids if vid not in processed_videos]
    logger.info(f'🎥 After removing processed videos: {len(video_ids)} remaining.')
    
    # Limit number of videos if specified
    if max_videos and len(video_ids) > max_videos:
        video_ids = video_ids[:max_videos]
    
    # Initialize or check if file exists
    file_exists = os.path.exists(OUTPUT_FILE)
    
    if not file_exists:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(DELIMITER.join(['videoId', 'username', 'comment', 'likeCount', 'publishedAt']) + '\n')
    
    # Process each video
    for idx, vid in enumerate(video_ids):
        logger.info(f'🔎 Processing video: {vid} ({idx+1}/{len(video_ids)})')
        
        try:
            # Check if we've reached quota limit for all keys
            if all(usage >= QUOTA_LIMIT for usage in api_usage.values()):
                logger.error("All API keys have reached quota limit. Stopping.")
                break
            
            # Fetch comments
            comments = fetch_comments(vid, max_comments_per_video)
            
            if comments:
                # Process and save comments
                df = pd.DataFrame(comments)
                df['comment'] = df['comment'].apply(clean_text)
                df['username'] = df['username'].apply(clean_text)
                
                # Save to CSV
                df.to_csv(OUTPUT_FILE, mode='a', index=False, sep=DELIMITER, header=False)
                logger.info(f'💾 Saved {len(df)} comments from video {vid}')
            
            # Mark video as processed
            save_processed_video(vid)
            
            # Add delay between requests
            time.sleep(0.5 + random.random() * 0.5)  # 0.5-1.0 seconds
            
        except Exception as e:
            if "quota exceeded" in str(e).lower():
                logger.warning(f"API quota issue detected, continuing with next API key")
                continue
            else:
                logger.error(f"⚠️ Error with {vid}: {e}")
    
    # Print API usage summary
    logger.info("API Usage Summary:")
    for key, usage in api_usage.items():
        logger.info(f"API key {key[-8:]}: {usage}/{QUOTA_LIMIT} units ({usage/QUOTA_LIMIT*100:.1f}%)")

if __name__ == '__main__':
    # Configure more conservative defaults
    MAX_VIDEOS = 100  # Limit total videos processed
    MAX_COMMENTS_PER_VIDEO = 50  # Limit comments per video
    
    logger.info("Starting YouTube comment scraper with quota optimization")
    scrape_comments(max_videos=MAX_VIDEOS, max_comments_per_video=MAX_COMMENTS_PER_VIDEO)
    logger.info("Scraping completed")