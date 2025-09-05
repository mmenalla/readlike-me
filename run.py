import argparse
import os
from dotenv import load_dotenv
from pipelines.goodreads_pipeline import goodreads_pipeline
from pipelines.ids_pipeline import ids_pipeline
from pipelines.rag_pipeline import rag_pipeline


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run Goodreads pipeline")
    parser.add_argument("--user_id", type=str, default=os.getenv("GOODREADS_USER_ID"),
                        help="Goodreads user ID (default: from .env)")
    args = parser.parse_args()

    if not args.user_id:
        raise ValueError("No user_id provided via argument or .env file.")

    # goodreads_pipeline(user_id=args.user_id)
    # rag_pipeline()
    ids_pipeline()

if __name__ == "__main__":
    main()
