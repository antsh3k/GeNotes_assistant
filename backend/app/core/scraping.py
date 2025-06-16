"""
Web scraping functionality for the application.
"""
import logging
import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)

def fetch_article(url: str) -> Dict[str, str]:
    """
    Fetch and parse an article from a given URL.
    
    Args:
        url: The URL of the article to fetch
        
    Returns:
        Dictionary containing the article data or error information
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract title - try to find the main heading
        title = soup.find("h1")
        if not title:
            title = soup.title.string if soup.title else "Untitled Document"
        else:
            title = title.text.strip()
        
        # Extract main content
        article = soup.find("article") or soup.find("main") or soup.find("div", class_=lambda x: x and "content" in x.lower())
        
        if article:
            # Remove script and style elements
            for script in article(["script", "style", "nav", "footer", "header", "aside"]):
                script.decompose()
            
            # Get text with proper spacing
            text = "\n".join(p.get_text().strip() for p in article.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"]))
        else:
            # Fallback to body text if no article/main/content div found
            text = soup.get_text()
        
        return {
            "url": url,
            "title": title,
            "content": text.strip(),
            "status": "success"
        }
    except requests.RequestException as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return {
            "url": url,
            "title": "Error",
            "content": f"Error fetching content: {str(e)}",
            "status": "error"
        }

def save_scraped_content(content: Dict, output_dir: Path) -> Optional[Path]:
    """
    Save scraped content to a file.
    
    Args:
        content: Dictionary containing the content to save
        output_dir: Directory to save the content to
        
    Returns:
        Path to the saved file or None if saving failed
    """
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate a filename from the URL
        url = content.get('url', 'untitled')
        filename = "".join(c if c.isalnum() else "_" for c in url)
        filename = f"scraped_{filename[:50]}.json"
        filepath = output_dir / filename
        
        # Save the content as JSON
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
            
        return filepath
    except Exception as e:
        logger.error(f"Error saving scraped content: {str(e)}")
        return None
