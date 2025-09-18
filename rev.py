# Verilog GitHub Repository Scraper using GitHub API

import requests
import pandas as pd
import time

# Replace with your GitHub personal access token (for higher rate limits)
GITHUB_TOKEN = "your_personal_access_token"
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

# Search query parameters
SEARCH_QUERY = "verilog language:Verilog"
BASE_URL = "https://api.github.com/search/repositories"
PARAMS = {
    "q": SEARCH_QUERY,
    "sort": "stars",
    "order": "desc",
    "per_page": 100,
    "page": 1
}

# List to store repository data
data = []

# Number of pages to scrape (GitHub search API max = 1000 results)
MAX_PAGES = 10

print("Starting data collection from GitHub...")

for page in range(1, MAX_PAGES + 1):
    PARAMS["page"] = page
    print(f"Fetching page {page}...")
    response = requests.get(BASE_URL, headers=HEADERS, params=PARAMS)

    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        break

    items = response.json().get("items", [])
    if not items:
        print("No more results.")
        break

    for repo in items:
        data.append({
            "name": repo["name"],
            "full_name": repo["full_name"],
            "description": repo.get("description"),
            "stars": repo["stargazers_count"],
            "forks": repo["forks_count"],
            "language": repo.get("language"),
            "html_url": repo["html_url"],
            "created_at": repo["created_at"],
            "updated_at": repo["updated_at"]
        })

    time.sleep(2)  # Respect API rate limits

# Convert to DataFrame and save
if data:
    df = pd.DataFrame(data)
    df.to_csv("verilog_github_repositories.csv", index=False)
    print("✅ Data saved to 'verilog_github_repositories.csv'")
else:
    print("⚠️ No data collected.")