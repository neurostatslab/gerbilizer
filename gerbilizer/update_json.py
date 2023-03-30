"""
Helper script to update parameters in a JSON file with new values from another.
"""

import argparse
import json


def read_json(path) -> dict:
    with open(path) as f:
        return json.load(f)


def update_json(original: dict, changes: dict, outpath: str):
    # apply the changes and save to json
    original.update(changes)
    with open(outpath, "w") as f:
        json.dump(original, f)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Update parameters in a JSON file with new values from another."
    )
    p.add_argument("original", type=str)
    p.add_argument("changes", type=str)
    p.add_argument("outpath", type=str)

    args = p.parse_args()

    original = read_json(args.original)
    changes = read_json(args.changes)

    update_json(original, changes, args.outpath)
