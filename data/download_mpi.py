"""
download_mpi.py -- Minimal MPI-INF-3DHP annotation downloader.

Downloads ONLY the annot.mat files (3D joint annotations) for each
subject/sequence -- skips all video files.  Total download: ~150 MB.

Usage:
    python data/download_mpi.py --dest data/mpi_inf_3dhp
    python data/download_mpi.py --dest data/mpi_inf_3dhp --subjects 1 2 3 4 5 6
"""

import argparse
import os
import urllib.request

BASE_URL = "http://gvv.mpi-inf.mpg.de/3dhp-dataset"

# Files we actually need -- annot.mat only (no videos)
ANNOT_FILES = ["annot.mat"]


def download_file(url: str, dest_path: str) -> bool:
    """Download url -> dest_path with a simple progress indicator."""
    if os.path.exists(dest_path):
        print(f"  [skip] already exists: {dest_path}")
        return True

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"  Downloading {url} ...")

    try:
        def _progress(block_num, block_size, total_size):
            if total_size > 0:
                pct = min(block_num * block_size * 100 / total_size, 100)
                print(f"\r    {pct:5.1f}%", end="", flush=True)

        urllib.request.urlretrieve(url, dest_path, reporthook=_progress)
        print()   # newline after progress
        return True
    except Exception as e:
        print(f"\n  [ERROR] Failed to download {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download MPI-INF-3DHP annotation files (annot.mat only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dest",     type=str, default="data/mpi_inf_3dhp",
                        help="Destination directory")
    parser.add_argument("--subjects", nargs="+", type=int,
                        default=[1, 2, 3, 4, 5, 6],
                        help="Subject numbers to download (1-6)")
    parser.add_argument("--seqs",     nargs="+", type=int,
                        default=[1, 2],
                        help="Sequence numbers to download (1 and/or 2)")
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    total = ok = 0

    for subj in args.subjects:
        for seq in args.seqs:
            print(f"\nSubject S{subj} / Seq{seq}")
            for fname in ANNOT_FILES:
                url       = f"{BASE_URL}/S{subj}/Seq{seq}/{fname}"
                dest_path = os.path.join(args.dest, f"S{subj}", f"Seq{seq}", fname)
                total += 1
                if download_file(url, dest_path):
                    ok += 1

    print(f"\nDone: {ok}/{total} files downloaded to '{args.dest}'")
    if ok == total:
        print("All annotation files ready.")
        print(f"\nNow train with:")
        print(f"  python train.py --dataset mpi --data_root {args.dest} --epochs 50")
    else:
        print(f"{total - ok} files failed. Check your internet connection or")
        print("make sure you have accepted the dataset licence at:")
        print("  https://vcai.mpi-inf.mpg.de/3dhp-dataset/")


if __name__ == "__main__":
    main()
