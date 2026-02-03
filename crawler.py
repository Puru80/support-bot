import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time


class SupportBotCrawler:
    def __init__(self, base_url, max_pages=10):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.max_pages = max_pages
        self.visited = set()
        self.to_visit = [base_url]
        self.scraped_data = []

    def is_valid(self, url):
        parsed = urlparse(url)
        return parsed.netloc == self.domain and url not in self.visited

    def clean_text(self, soup):
        # Remove script and style elements
        for script_or_style in soup(["script", "style", "nav", "footer", "header"]):
            script_or_style.decompose()

        # Focus on content-heavy tags
        chunks = []
        for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'li']):
            text = tag.get_text(strip=True)
            if len(text) > 20:  # Ignore tiny fragments
                chunks.append(text)

        return "\n".join(chunks)

    def crawl(self):
        while self.to_visit and len(self.visited) < self.max_pages:
            url = self.to_visit.pop(0)
            if url in self.visited:
                continue

            print(f"Crawling: {url}")
            try:
                response = requests.get(url, timeout=5)
                if response.status_code != 200:
                    continue

                soup = BeautifulSoup(response.text, 'html.parser')
                self.visited.add(url)

                # Extract content
                content = self.clean_text(soup)
                self.scraped_data.append({"url": url, "content": content})

                # Find internal links
                for link in soup.find_all('a', href=True):
                    full_url = urljoin(self.base_url, link['href'])
                    if self.is_valid(full_url):
                        self.to_visit.append(full_url)

                time.sleep(1)  # Be polite to the server
            except Exception as e:
                print(f"Failed to crawl {url}: {e}")

        return self.scraped_data