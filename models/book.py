class Book:
    def __init__(self, title, author, url, book_description=None, author_description=None, passages=[]):
        self.title = title
        self.author = author
        self.url = url
        self.book_description = book_description
        self.author_description = author_description
        self.passages = passages

    def to_dict(self):
        return {
            "title": self.title,
            "author": self.author,
            "url": self.url,
            "book_description": self.book_description,
            "author_description": self.author_description,
            "passages": []
        }
