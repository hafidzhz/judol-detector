# scraper.py
import requests
import pandas as pd
import time
import sys

API_KEY = ''  # 🔐 Replace with your actual API key
OUTPUT_FILE = 'comments_single_video.csv'
DELIMITER = '¤'

def fetch_comments(video_id, api_key):
    comments = []
    url = 'https://www.googleapis.com/youtube/v3/commentThreads'
    params = {
        'part': 'snippet',
        'videoId': video_id,
        'maxResults': 100,
        'key': api_key,
        'textFormat': 'plainText'
    }

    while True:
        response = requests.get(url, params=params)
        data = response.json()

        for item in data.get('items', []):
            text = item['snippet']['topLevelComment']['snippet']['textOriginal']
            comments.append(clean_text(text))

        if 'nextPageToken' in data:
            params['pageToken'] = data['nextPageToken']
            time.sleep(0.5)
        else:
            break

    return comments

def clean_text(text):
    return str(text).replace('\n', ' ').replace('\r', ' ').strip()

def scrape_single_video(video_id):
    print(f'🎥 Fetching comments for video ID: {video_id}')
    try:
        comments = fetch_comments(video_id, API_KEY)
        if comments:
            df = pd.DataFrame(comments, columns=['comment'])
            df.to_csv(OUTPUT_FILE, index=False)
            print(f'✅ Saved {len(df)} comments to "{OUTPUT_FILE}"')
        else:
            print('⚠️ No comments found for this video.')
    except Exception as e:
        print(f'❌ Error fetching comments: {e}')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("⚠️ Usage: python scraper.py VIDEO_ID")
    else:
        video_id = sys.argv[1]
        scrape_single_video(video_id)
