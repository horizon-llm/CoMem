#!/usr/bin/env python3
"""
Create a mid-size WebShop subset to reduce memory footprint.

Outputs:
  <out_dir>/items_shuffle_<N>.json
  <out_dir>/items_ins_v2_<N>.json
  <out_dir>/items_human_ins_<N>.json   (if --human provided and entries exist)
  <out_dir>/documents_<N>.jsonl        (for building a smaller search index)

Usage example:
python make_subset.py \
    -n 10000 --seed 42 \
    --products webshop/data/items_shuffle.json \
    --attrs webshop/data/items_ins_v2.json \
    --human webshop/data/items_human_ins.json \
    --out-dir webshop/data/subsets/10k

Minimal dependencies: only standard library.
"""

import argparse
import json
import random
import os
from pathlib import Path
from typing import List, Dict, Any, Set

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def select_products(products: List[Dict[str, Any]], n: int, seed: int, mode: str) -> List[Dict[str, Any]]:
    """
    mode = 'first' (take first n; relies on prior shuffle),
           'sample' (uniform sample without replacement).
    """
    if n >= len(products):
        return products

    if mode == 'first':
        return products[:n]
    elif mode == 'sample':
        rng = random.Random(seed)
        return rng.sample(products, n)
    else:
        raise ValueError(f"Unknown selection mode: {mode}")

def filter_attrs(full_attr: Dict[str, Any], keep_asins: Set[str]) -> Dict[str, Any]:
    return {asin: v for asin, v in full_attr.items() if asin in keep_asins}

def filter_human(human_attr: Dict[str, Any],
                 keep_asins: Set[str],
                 drop_empty: bool) -> Dict[str, Any]:
    out = {}
    for asin, inst_list in human_attr.items():
        if asin not in keep_asins:
            continue
        if not isinstance(inst_list, list):
            continue
        new_list = []
        for inst in inst_list:
            # Typical fields: instruction, instruction_attributes, instruction_options
            atts = inst.get("instruction_attributes", [])
            if drop_empty and (not atts):
                continue
            new_list.append(inst)
        if new_list:
            out[asin] = new_list
    return out

def write_subset_files(sub_products: List[Dict[str, Any]],
                       sub_attrs: Dict[str, Any],
                       sub_human: Dict[str, Any],
                       out_dir: Path,
                       n: int,
                       write_human: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    prod_path = out_dir / f"items_shuffle_{n}.json"
    attr_path = out_dir / f"items_ins_v2_{n}.json"
    human_path = out_dir / f"items_human_ins_{n}.json"

    with open(prod_path, "w") as f:
        json.dump(sub_products, f)
    with open(attr_path, "w") as f:
        json.dump(sub_attrs, f)
    if write_human and sub_human:
        with open(human_path, "w") as f:
            json.dump(sub_human, f)

    print(f"Wrote products: {prod_path}")
    print(f"Wrote synthetic attrs: {attr_path}")
    if write_human and sub_human:
        print(f"Wrote human instructions: {human_path}")

def build_documents_jsonl(sub_products: List[Dict[str, Any]], out_dir: Path, n: int):
    """
    Creates a smaller documents_<N>.jsonl suitable for a down-sized Lucene index.
    Each line: {"id": <asin>, "contents": <text blob>}
    Contents can be tuned; we keep title + description + bullet points for retrievability.
    """
    doc_path = out_dir / f"documents_{n}.jsonl"
    with open(doc_path, "w") as f:
        for p in sub_products:
            asin = p.get("asin")
            title = p.get("name") or p.get("Title") or ""
            desc = p.get("full_description") or p.get("Description") or ""
            bullets = p.get("small_description") or p.get("BulletPoints") or []
            if isinstance(bullets, list):
                bullets_text = " ".join(str(b) for b in bullets)
            else:
                bullets_text = str(bullets)
            contents = " ".join(filter(None, [title, desc, bullets_text]))
            # Minimal JSON doc
            doc = {"id": asin, "contents": contents}
            f.write(json.dumps(doc) + "\n")
    print(f"Wrote search documents: {doc_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", type=int, required=True,
                    help="Target number of products in subset.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for sampling (used if mode=sample).")
    ap.add_argument("--mode", choices=["first", "sample"], default="first",
                    help="Selection strategy. 'first' takes first N (files are pre-shuffled). 'sample' draws random subset.")
    ap.add_argument("--products", required=True,
                    help="Path to full items_shuffle.json (full dataset).")
    ap.add_argument("--attrs", required=True,
                    help="Path to full items_ins_v2.json (synthetic attribute/instruction file).")
    ap.add_argument("--human",
                    help="Path to items_human_ins.json (optional, for human goals).")
    ap.add_argument("--drop-empty-human", action="store_true",
                    help="Drop human instructions with empty instruction_attributes.")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory for subset files.")
    ap.add_argument("--no-docs", action="store_true",
                    help="Skip writing documents_<N>.jsonl for search indexing.")
    args = ap.parse_args()

    # Load full data
    print("Loading full product file...")
    products = load_json(args.products)
    print(f"Loaded {len(products):,} products.")

    print("Loading full synthetic attribute file...")
    full_attrs = load_json(args.attrs)
    print(f"Loaded synthetic attrs for {len(full_attrs):,} ASINs.")

    human_data = {}
    if args.human:
        print("Loading full human instruction file...")
        if os.path.exists(args.human):
            human_data = load_json(args.human)
            print(f"Loaded human instructions for {len(human_data):,} ASINs.")
        else:
            print(f"[WARN] Human file {args.human} not found. Continuing without human data.")

    # Select subset
    print(f"Selecting {args.num} products (mode={args.mode})...")
    subset_products = select_products(products, args.num, args.seed, args.mode)
    print(f"Subset size: {len(subset_products):,}")

    keep_asins = {p["asin"] for p in subset_products if "asin" in p}

    # Filter attribute/synthetic instruction dictionary
    sub_attrs = filter_attrs(full_attrs, keep_asins)
    print(f"Synthetic attr entries kept: {len(sub_attrs):,}")

    # Filter human instructions if provided
    sub_human = {}
    if human_data:
        sub_human = filter_human(human_data, keep_asins, args.drop_empty_human)
        print(f"Human instruction entries kept: {len(sub_human):,}")

    out_dir = Path(args.out_dir)
    write_subset_files(subset_products, sub_attrs, sub_human, out_dir, args.num, write_human=bool(human_data))

    if not args.no_docs:
        build_documents_jsonl(subset_products, out_dir, args.num)

    print("Done. To use this subset, point file_path / attr_path to the new files or update DEFAULT_* in utils.py.")

if __name__ == "__main__":
    main()