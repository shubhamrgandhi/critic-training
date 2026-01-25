#!/usr/bin/env python3
from datasets import load_dataset

def main():
    # 1) load SWE-bench Lite "dev" (HF only exposes a single split there, so we take the default)
    lite = load_dataset("princeton-nlp/SWE-bench_Lite")["dev"] \
        if "dev" in load_dataset("princeton-nlp/SWE-bench_Lite") \
        else load_dataset("princeton-nlp/SWE-bench_Lite")["test"]

    # 2) load SWE-bench Verified (has a test split) :contentReference[oaicite:0]{index=0}
    verified = load_dataset("princeton-nlp/SWE-bench_Verified")["test"]

    # grab ids
    lite_ids = {row["instance_id"] for row in lite}
    verified_ids = {row["instance_id"] for row in verified}

    overlap = lite_ids & verified_ids

    print(f"SWE-bench Lite size: {len(lite_ids)}")
    print(f"SWE-bench Verified size: {len(verified_ids)}")
    print(f"Overlap: {len(overlap)}")
    print()

    if overlap:
        print("IDs present in both:")
        for _id in sorted(overlap):
            print(_id)

if __name__ == "__main__":
    main()
