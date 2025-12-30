import requests
import time
import hashlib
from datetime import datetime

# --- CONFIGURATION ---
URLS_TO_MONITOR = [
    "http://www.blackfile-index.org/",
    "http://www.blackfile-index.org/script.js",
]

CHECK_INTERVAL = 30  # Seconds between checks

# Paste your Webhook URL here
DISCORD_WEBHOOK_URL = "Webhook_Here"


# ---------------------

def send_discord_message(content):
    data = {
        "content": content,
        "username": "WebsiteUpdates"
    }
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=data)
    except Exception as e:
        print(f"[{datetime.now()}] Failed to send Discord message: {e}")


def send_startup_message():
    """Sends a test message to confirm the script is running."""
    msg = (
        "✅ **Monitor Started Successfully!**\n"
        f"Time: {datetime.now()}\n"
    )
    send_discord_message(msg)
    print(f"[{datetime.now()}] Startup test message sent.")


def get_website_hash(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/58.0.3029.110 Safari/537.3'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return hashlib.sha256(response.text.encode('utf-8')).hexdigest()
    except requests.exceptions.RequestException as e:
        print(f"[{datetime.now()}] Error fetching {url}: {e}")
        return None


def monitor_websites():
    print(f"[{datetime.now()}] Initializing monitor...")

    # 1. Send the test message immediately
    send_startup_message()

    saved_hashes = {}

    # 2. Get initial state
    for url in URLS_TO_MONITOR:
        initial_hash = get_website_hash(url)
        if initial_hash:
            saved_hashes[url] = initial_hash
            print(f"Successfully indexed: {url}")
        else:
            print(f"⚠️ Could not fetch {url} on startup.")

    print("--- Monitoring Loop Started ---")

    # 3. Continuous Loop
    while True:
        time.sleep(CHECK_INTERVAL)

        for url in URLS_TO_MONITOR:
            current_hash = get_website_hash(url)

            if current_hash is None:
                continue

            # If it's a new URL we haven't seen yet, save it and move on
            if url not in saved_hashes:
                saved_hashes[url] = current_hash
                continue

            # Check for changes
            if current_hash != saved_hashes[url]:
                print(f"[{datetime.now()}] 🚨 CHANGE DETECTED on {url}!")

                alert_text = f" **CHANGE DETECTED!** \n\nThe page {url} has changed."
                send_discord_message(alert_text)

                saved_hashes[url] = current_hash


if __name__ == "__main__":
    monitor_websites()