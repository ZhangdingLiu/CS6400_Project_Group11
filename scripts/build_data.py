import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import WikipediaDataLoader, build_data_files


def main():
    parser = argparse.ArgumentParser(description='Generate Wikipedia embeddings and metadata')
    parser.add_argument(
        '--n-samples',
        type=int,
        default=100000,
        help='Number of samples to generate (default: 100000, use 0 for all 224k)'
    )
    parser.add_argument(
        '--embedding-method',
        type=str,
        choices=['OpenAI', 'MiniLM', 'GTE-small'],
        default='OpenAI',
        help='Embedding method (default: OpenAI)'
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default='data_files',
        help='Output directory (default: data_files)'
    )

    args = parser.parse_args()

    # Convert 0 to None (means all data)
    n_samples = None if args.n_samples == 0 else args.n_samples

    print("="*60)
    print("Wikipedia Dataset Generator")
    print("="*60)
    print(f"Embedding method: {args.embedding_method}")
    print(f"Output directory: {args.out_dir}")

    if n_samples is None:
        print(f"Samples: ALL (224,482 records, ~1.3GB)")
        print("WARNING: This will take ~10-15 minutes and use significant memory")
    else:
        size_mb = n_samples * 1536 * 4 / (1024**2)
        print(f"Samples: {n_samples:,} records (~{size_mb:.0f}MB)")

    print("="*60)

    # Load data
    loader = WikipediaDataLoader(embedding_method=args.embedding_method, text_field="body")

    # Build data files
    print("\nGenerating data...")
    emb_path, meta_path = build_data_files(loader, out_dir=args.out_dir, n_samples=n_samples)

    print("\n" + "="*60)
    print("SUCCESS!")
    print("="*60)
    print(f"Embeddings: {emb_path}")
    print(f"Metadata:   {meta_path}")

    # Show file sizes
    emb_size = os.path.getsize(emb_path) / (1024**2)
    meta_size = os.path.getsize(meta_path) / (1024**2)
    print(f"\nFile sizes:")
    print(f"  {emb_path}: {emb_size:.2f} MB")
    print(f"  {meta_path}: {meta_size:.2f} MB")
    print(f"  Total: {emb_size + meta_size:.2f} MB")


if __name__ == "__main__":
    main()