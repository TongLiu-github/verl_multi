#!/usr/bin/env python3
"""
Script to check the structure of data files for multi-prompt support
"""

import argparse
import pandas as pd
import datasets


def check_data_structure(data_path):
    """Check if the data file contains the required fields for multi-prompt."""
    print(f"Checking data file: {data_path}")
    
    # Load the dataset
    try:
        dataset = datasets.load_dataset("parquet", data_files=data_path)
        data = dataset["train"]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False
    
    print(f"Dataset size: {len(data)}")
    print(f"Columns: {data.column_names}")
    
    # Check if extra_info exists
    if "extra_info" not in data.column_names:
        print("❌ 'extra_info' column not found")
        return False
    
    print("✅ 'extra_info' column found")
    
    # Check a few samples for extra_info.question
    has_question = False
    for i in range(min(5, len(data))):
        sample = data[i]
        extra_info = sample.get("extra_info", {})
        if isinstance(extra_info, dict) and "question" in extra_info:
            has_question = True
            print(f"✅ Sample {i}: Found 'question' in extra_info")
            print(f"   Question: {extra_info['question'][:100]}...")
        else:
            print(f"❌ Sample {i}: No 'question' in extra_info")
            print(f"   extra_info keys: {list(extra_info.keys()) if isinstance(extra_info, dict) else 'not a dict'}")
    
    if has_question:
        print("\n✅ Data file is ready for multi-prompt rollout!")
        return True
    else:
        print("\n❌ Data file needs to be regenerated with 'question' field in extra_info")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to the parquet data file")
    args = parser.parse_args()
    
    check_data_structure(args.data_path) 