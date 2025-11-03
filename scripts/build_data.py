from data import WikipediaDataLoader, build_data_files

def main():
    # embedding_method in "OpenAI", "MiniLM", or  "GTE-small"
    loader = WikipediaDataLoader(embedding_method="OpenAI", text_field="body")
    emb_path, meta_path = build_data_files(loader, out_dir="data_files", n_samples=10000)
    print("Saved:", emb_path, "and", meta_path)

if __name__ == "__main__":
    main()