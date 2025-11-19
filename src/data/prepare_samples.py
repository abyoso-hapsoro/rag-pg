import os
import contextlib
from datasets import load_dataset


def sample_ag_news(n: int = 100, seed: int = 123) -> None:
    """
    Download AG News dataset, sample n rows and export to CSV.

    Args:
        n (int, optional):
            Number of samples (default: 100).
        seed (int, optional):
            Random seed for reproducibility (default: 123).
    """

    # Suppress all Hugging Face outputs
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        ds = load_dataset("sentence-transformers/agnews", split="train")

    # Convert to pandas
    df = ds.to_pandas()

    # Standardize column name to match schema
    df = df.rename(columns={"description": "content"})

    # Sample n rows
    samples = df.sample(n=n, random_state=seed).reset_index(drop=True)

    # Export to CSV
    output_path = "src/data/AGNews-100.csv"
    samples.to_csv(
        output_path,
        index=False,
        quoting=1,
        encoding="utf-8"
    )
    print(f"Prepared {n} samples to '{output_path}'")


if __name__ == "__main__":
    sample_ag_news()
