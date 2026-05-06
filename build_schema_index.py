from src.schema_indexer import build_or_refresh_vectorstore


if __name__ == "__main__":
    build_or_refresh_vectorstore()
    print("Schema vector index created/refreshed.")
