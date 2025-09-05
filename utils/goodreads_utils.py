from datetime import datetime

import requests
from bs4 import BeautifulSoup
from zenml import get_step_context

HEADERS = {"User-Agent": "Mozilla/5.0"}

def get_soup(url: str) -> BeautifulSoup:
    res = requests.get(url, headers=HEADERS)
    res.raise_for_status()
    return BeautifulSoup(res.text, "html.parser")


def get_soup(url: str) -> BeautifulSoup:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def get_book_and_author_description(book_url: str) -> tuple[str | None, str | None]:
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(book_url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # --- BOOK DESCRIPTION ---
    book_desc_tag = soup.select_one(
        "div.BookPageMetadataSection__description span.Formatted"
    )
    book_description = book_desc_tag.get_text(strip=True) if book_desc_tag else None

    # --- AUTHOR DESCRIPTION ---
    author_desc_tag = soup.select_one(
        "div.AuthorPreview div.TruncatedContent__text--medium span.Formatted"
    )
    author_description = author_desc_tag.get_text(strip=True) if author_desc_tag else None

    return book_description, author_description

def crawl_goodreads_read_shelf(user_id: str, shelf: str = "read", page: int = 1) -> list[dict[str, str | None]]:
    url = f"https://www.goodreads.com/review/list/{user_id}?ref=nav_mybooks&shelf={shelf}&per_page=100&page={page}"
    soup = get_soup(url)

    rows = soup.select("tr.bookalike.review")
    books = []

    for row in rows:
        title_tag = row.select_one("td.field.title a")
        author_tag = row.select_one("td.field.author a")

        title = title_tag.text.strip() if title_tag else None
        author = author_tag.text.strip() if author_tag else None
        book_url = f"https://www.goodreads.com{title_tag['href']}" if title_tag else None

        books.append({"title": title, "author": author, "url": book_url})

    metadata = add_to_metadata(
        {
            "goodreads": {
                "successful": 1,
                "total": 1,
                "books_listed": len(books)
            }
        },
        user_id,
        True,
        len(books)
    )
    metadata["timestamp"] = datetime.now().isoformat()
    step_context = get_step_context()
    step_context.add_output_metadata(output_name="output", metadata=metadata)

    return books

def add_to_metadata(metadata: dict, domain: str, successful_crawl: bool, num_books: int = 0) -> dict:
    if domain not in metadata:
        metadata[domain] = {}
    metadata[domain]["successful"] = metadata[domain].get("successful", 0) + (1 if successful_crawl else 0)
    metadata[domain]["total"] = metadata[domain].get("total", 0) + 1
    metadata[domain]["books_listed"] = metadata[domain].get("books_listed", 0) + num_books
    return metadata
