"""
Scrape Hinglish comments from YouTube videos.
Saves to data/raw/hinglish_raw.jsonl (one JSON object per line).
"""

import json
import os
import time
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

# YouTube video IDs — mix of IPL, Bollywood, Hindi news (rich Hinglish source)
VIDEO_IDS = [
    "laQL2nNqw9M",  # Lallantop - Satinder Sartaaj interview
    "Y0Af7MjC9yA",  # ABP News - Bengal election
    "IDBps4ww8-s",  # MissMalini - Bollywood industry
    "irZWvVLoUYg",  # CriAddict - Hindi commentary
    "bfDdzX4gXbg",  # Khan Sir - UPSC
]

# Minimum comment length to filter out single-word reactions
MIN_LENGTH = 20
MAX_COMMENTS_PER_VIDEO = 500


def get_comments(youtube, video_id):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText",
    )
    while request and len(comments) < MAX_COMMENTS_PER_VIDEO:
        try:
            response = request.execute()
        except Exception as e:
            print(f"  Skipping video {video_id}: {e}")
            break

        for item in response.get("items", []):
            text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            if len(text) >= MIN_LENGTH:
                comments.append({
                    "text": text,
                    "video_id": video_id,
                    "source": "youtube"
                })

        request = youtube.commentThreads().list_next(request, response)
        time.sleep(0.5)  # stay within quota

    return comments


def main():
    os.makedirs("data/raw", exist_ok=True)
    out_path = "data/raw/hinglish_raw.jsonl"

    youtube = build("youtube", "v3", developerKey=API_KEY)
    all_comments = []

    for vid in VIDEO_IDS:
        print(f"Scraping video: {vid}")
        comments = get_comments(youtube, vid)
        print(f"  Collected {len(comments)} comments")
        all_comments.extend(comments)
        time.sleep(1)

    # Deduplicate by text
    seen = set()
    unique = []
    for c in all_comments:
        if c["text"] not in seen:
            seen.add(c["text"])
            unique.append(c)

    with open(out_path, "w", encoding="utf-8") as f:
        for c in unique:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(unique)} unique comments saved to {out_path}")


if __name__ == "__main__":
    main()
