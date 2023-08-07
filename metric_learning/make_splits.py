#!/usr/bin/env python3

import argparse
import time
import pandas as pd
from typing import List, Tuple

from gram2vec import vectorizer


def measure_time(func):
    """Debugging function for measuring function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.6f} seconds.")
        return result
    return wrapper

def load_data(path:str) -> pd.DataFrame:
    df = vectorizer.load_jsonlines(path)
    df["authorIDs"] = df["authorIDs"].apply(lambda x: "".join(x))
    return df

def get_unique_author_ids(df:pd.DataFrame) -> List[str]:
    return df["authorIDs"].unique().tolist()

def split_author_ids(ids:List[str], num_train:int, num_dev:int, num_test:int) -> Tuple[List[str]]:
    return ids[:num_train], ids[num_train:num_dev], ids[num_dev:num_test]

@measure_time
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-n",
                        "--dataset_name",
                        default="hrs_release_May23DryRun")
    parser.add_argument("-d",
                        "--dataset_dir",
                        default="../data/pan22/preprocessed/pan22_preprocessed.jsonl")
    
    
    args = parser.parse_args()
    data = load_data(args.dataset_dir)
    author_ids = get_unique_author_ids(data)
    
    train, dev, test = split_author_ids(author_ids)
    import ipdb;ipdb.set_trace()
    