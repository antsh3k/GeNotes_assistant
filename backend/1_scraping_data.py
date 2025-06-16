#################################################################################################################################################################
###############################   1.  IMPORTING MODULES AND INITIALIZING VARIABLES   ############################################################################
#################################################################################################################################################################

from dotenv import load_dotenv
import os
import requests
import json
import pandas as pd
from bs4 import BeautifulSoup
import glob

pd.options.mode.chained_assignment = None

load_dotenv()

#################################################################################################################################################################
###############################   CONFIGURATION (USE .env OR HARDCODED PATHS)   #################################################################################
#################################################################################################################################################################

DISEASE_LIST_FILE = os.getenv('DISEASE_LIST_FILE')
SNAPSHOT_STORAGE_FILE = os.getenv('SNAPSHOT_STORAGE_FILE')
DATASET_STORAGE_FOLDER = os.getenv('DATASET_STORAGE_FOLDER')
BASE_URL = "https://www.genomicseducation.hee.nhs.uk/genotes/knowledge-hub/"

#################################################################################################################################################################
###############################   2.  IF SnapshotID IS NOT SET IN .TXT FILE, START SCRAPING FROM GENOTES   #####################################################
#################################################################################################################################################################

file_exists = os.path.isfile(SNAPSHOT_STORAGE_FILE)

# Check if snapshot already exists
file_exists = os.path.isfile(SNAPSHOT_STORAGE_FILE)

def fetch_article(slug):
    full_url = BASE_URL + slug + "/"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(full_url, headers=headers)

    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.find("h1").text.strip() if soup.find("h1") else slug
    body = soup.find("article")

    text_content = body.get_text(separator="\n", strip=True) if body else "No content found"
    
    return {
        "slug": slug,
        "title": title,
        "url": full_url,
        "content": text_content
    }

if not file_exists:
    with open(DISEASE_LIST_FILE, "r") as f:
        slugs = [line.strip() for line in f if line.strip()]

    results = []

    for slug in slugs:
        print(f"🔍 Fetching: {slug}")
        article_data = fetch_article(slug)
        if article_data:
            results.append(article_data)
        else:
            print(f"⚠️ Failed to fetch: {slug}")

    # Save snapshot ID (fake one based on count)
    snapshot_id = "genotes_" + str(len(results))
    with open(SNAPSHOT_STORAGE_FILE, "w") as f:
        f.write(snapshot_id)

    # Save results
    if not os.path.exists(DATASET_STORAGE_FOLDER):
        os.makedirs(DATASET_STORAGE_FOLDER)

    with open(DATASET_STORAGE_FOLDER + "data.txt", "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Snapshot created with {len(results)} entries.")

else:
    # TODO: Check if articles have already been fetched. Skip and only fetch new articles. right now it fetches all articles again.
    print(f"📄 Snapshot file exists. Re-fetching data...")

    # Clear old data
    files = glob.glob(DATASET_STORAGE_FOLDER + "*")
    for f in files:
        os.remove(f)

    with open(DISEASE_LIST_FILE, "r") as f:
        slugs = [line.strip() for line in f if line.strip()]

    results = []

    for slug in slugs:
        print(f"🔁 Re-fetching: {slug}")
        article_data = fetch_article(slug)
        if article_data:
            results.append(article_data)
        else:
            print(f"⚠️ Failed to fetch: {slug}")

    with open(DATASET_STORAGE_FOLDER + "data.txt", "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Refreshed dataset saved with {len(results)} entries.")
