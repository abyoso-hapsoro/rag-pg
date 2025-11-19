import os
import shutil


def delete_all_pycache(start_dir: str = "."):
    """
    Recursively delete all __pycache__ folders from start_dir.
    """
    for root, dirs, files in os.walk(start_dir, topdown=False):
        for d in dirs:
            if d == "__pycache__":
                full_path = os.path.join(root, d)
                print(f"Deleting: {full_path}")
                shutil.rmtree(full_path, ignore_errors=True)


if __name__ == "__main__":
    delete_all_pycache()
