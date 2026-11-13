import chromadb

client = chromadb.PersistentClient(path="data/chroma")

def search_guidelines(question, category="validation",
                      source=None, n_results=3):

    collection = client.get_collection(
        f"{category}_guidelines"
    )

    params = {
        "query_texts": [question],
        "n_results": n_results
    }

    if source:
        params["where"] = {"source": source}

    return collection.query(**params)

if __name__ == "__main__":
    # Test retrieval
    results = search_guidelines(
        "How are stock changes treated in food balance sheets?",
        "validation"
    )

    for text, metadata in zip(
        results["documents"][0],
        results["metadatas"][0]
    ):
        print("Source:", metadata["source"])
        print("Text:", text)
        print()
