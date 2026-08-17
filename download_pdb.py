"""
Download a protein structure from the RCSB Protein Data Bank.

Usage:
    python download_pdb.py 2a3d          # -> data/2a3d.pdb
    python download_pdb.py 1crn --out .  # -> ./1crn.pdb

Any 4-character PDB ID works; point main.py at the downloaded file with --pdb.
"""

import argparse
import os
import sys
import urllib.request

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def download(pdb_id, out_dir="data"):
    pdb_id = pdb_id.lower()
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, f"{pdb_id}.pdb")
    url = RCSB_URL.format(pdb_id=pdb_id)

    print(f"Downloading {url} ...")
    try:
        urllib.request.urlretrieve(url, dest)
    except urllib.error.HTTPError as exc:
        sys.exit(f"Failed to download '{pdb_id}': {exc}. Check the PDB ID.")
    print(f"Saved to {dest}")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a PDB file from RCSB.")
    parser.add_argument("pdb_id", help="4-character PDB ID, e.g. 2a3d")
    parser.add_argument("--out", default="data", help="output directory (default: data)")
    args = parser.parse_args()
    download(args.pdb_id, args.out)
