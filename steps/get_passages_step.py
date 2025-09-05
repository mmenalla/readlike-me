from zenml import step
from typing import List, Dict
from utils.gutenberg_utils import get_gutenberg_passages
from utils.openai_utils import ask_llm_passages, ask_llm_plain


@step
def get_passages(
    books: List[Dict],
    count: int = 5,
    sentences_per_passage: int = 10
) -> List[Dict]:
    enriched_books = []
    for book in books:
        author = book.get("author", "")
        title = book.get("title", "")
        if not author or not title:
            print(f"Skipping book with missing author or title: {book}")
            continue

        author = str(author)
        title = str(title)
        passages = get_gutenberg_passages(
            author=author,
            title=title,
            count=count,
            sentences_per_passage=sentences_per_passage
        )

        if len(passages) == 0:
            print(f"No passages found for {author} - {title}")
            print("Generating from OpenAI...")
            passages = ask_llm_passages(
                author=author,
                title=title,
                num_passages=count,
                num_sentences=sentences_per_passage
            )

        book["passages"] = passages
        enriched_books.append(book)

    # Add GENERIC/no-style baseline passages
    print("Generating GENERIC baseline passages...")
    for i in range(2):
        generic_passages = ask_llm_plain(
            num_passages=count,
            num_sentences=sentences_per_passage
        )
        enriched_books.append({
            "author": "GENERIC",
            "title": "GENERIC",
            "passages": generic_passages
        })

    return enriched_books
