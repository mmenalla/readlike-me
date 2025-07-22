from zenml import step
from typing import List, Dict
from utils.goodreads_utils import crawl_goodreads_read_shelf

@step
def crawl_books(user_id: str) -> List[Dict]:
    return crawl_goodreads_read_shelf(user_id=user_id)
