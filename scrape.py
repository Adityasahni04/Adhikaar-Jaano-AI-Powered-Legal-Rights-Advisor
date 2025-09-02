import os
import json
import re
import requests
from typing import Dict, Any, List
from bs4 import BeautifulSoup
import google.generativeai as genai

# ✅ Configure Gemini
genai.configure(api_key="AIzaSyBfoib8_HBZ8fi7ccxz0Ruwd40_b1irhkw")

def clean_text(text: str) -> str:
    """Clean scraped text to remove extra spaces and junk."""
    text = re.sub(r'\s+', ' ', text)  # collapse spaces/newlines
    return text.strip()

def extract_main_content(soup: BeautifulSoup) -> str:
    """Extract main content from a legal webpage."""
    # Remove scripts, styles, nav, footer
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
        tag.extract()

    # Grab text from headings + paragraphs + articles
    content_blocks: List[str] = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "article", "section", "div"]):
        text = tag.get_text(separator=" ", strip=True)
        if text and len(text.split()) > 5:  # meaningful chunk
            content_blocks.append(text)

    return clean_text(" ".join(content_blocks))

def scrape_website(url: str, query: str = "") -> str:
    """Scrape and return cleaned content from a given URL, optionally filter by query."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200:
            return ""

        soup = BeautifulSoup(res.text, "html.parser")
        full_text = extract_main_content(soup)

        # Optional: keep only paragraphs relevant to query (for large IPC docs)
        if query:
            relevant = [p for p in full_text.split(". ") if query.lower() in p.lower()]
            if relevant:
                return clean_text(". ".join(relevant))

        return full_text[:5000]  # keep reasonable length
    except Exception as e:
        return ""

def search_duckduckgo(query: str) -> str:
    """Search DuckDuckGo and return first result URL."""
    try:
        res = requests.get("https://duckduckgo.com/html/",
                           params={"q": query},
                           headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        links = soup.select(".result__a")

        for a in links:
            href = a.get("href")
            if href.startswith("http") and not "duckduckgo" in href:
                return href
        return ""
    except:
        return ""

def get_answer(query: str) -> Dict[str, Any]:
    """Try scraping multiple results, else Gemini fallback."""

    # Step 1: Search top DuckDuckGo results
    res = requests.get("https://duckduckgo.com/html/",
                       params={"q": query + " site:indiacode.nic.in OR site:vakilno1.com OR site:legislative.gov.in"},
                       headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(res.text, "html.parser")
    links = [a.get("href") for a in soup.select(".result__a") if a.get("href", "").startswith("http")]

    scraped_content = []
    for url in links[:3]:  # try top 3 links
        content = scrape_website(url, query=query)
        if len(content) > 100:
            scraped_content.append((url, content))

    if scraped_content:
        # Combine best result
        best_url, best_text = scraped_content[0]
        return {
            "answer": best_text[:1200],
            "source": "scraping",
            "url": best_url
        }

    # Step 3: Gemini fallback
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(query)
    
    return {
        "answer": response.text,
        "source": "gemini",
        "url": None
    }


# Example usage
if __name__ == "__main__":
    query = "What to file Divorce?"
    result = get_answer(query)
    print(json.dumps(result, indent=2))
