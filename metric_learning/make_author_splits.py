#!/usr/bin/env python3

import argparse
import time
import pandas as pd
import json
from typing import List, Tuple, Dict

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
    """Splits the author ids into segments. Requires knowing roughly how many authors are in the dataset"""
    train = ids[:num_train]
    dev = ids[num_train:num_train+num_dev]
    test = ids[num_train+num_dev:num_train+num_dev+num_test]
    return train, dev, test

def get_author_entries(df:pd.DataFrame, author_id:str) -> pd.DataFrame:
    return df.loc[df["authorIDs"] == author_id]

def get_all_entries_from_authors(data:pd.DataFrame, authors:List[str]) -> List[Dict]:
    entries = []
    for author_id in authors:
        author_entries = get_author_entries(data, author_id)
        entries.extend([entry for entry in author_entries.to_dict(orient="records")])
    return entries

def save_to_jsonlines(data:List[Dict], path:str) -> None:
    with open(path, "w") as file:
        for item in data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')

@measure_time
def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-d",
                        "--dataset_dir",
                        default="../data/hrs_release_May23DryRun")
    
    parser.add_argument("--num_train",
                        type=int,
                        default=None)
    
    parser.add_argument("--num_dev",
                        type=int,
                        default=None)
    
    parser.add_argument("--num_test",
                        type=int,
                        default=None)
    args = parser.parse_args()
    
    num_train = args.num_train
    num_dev = args.num_dev
    num_test = args.num_test
    
    data = load_data(args.dataset_dir)
    author_ids = get_unique_author_ids(data)
    train, dev, test = split_author_ids(author_ids, 
                                        num_train, 
                                        num_dev, 
                                        num_test)
    
    train_entries = get_all_entries_from_authors(data, train)
    dev_entries = get_all_entries_from_authors(data, dev)
    test_entries = get_all_entries_from_authors(data, test)
    
    save_to_jsonlines(train_entries, "splits/hrs_release_May23DryRun/train/train.jsonl")
    save_to_jsonlines(dev_entries, "splits/hrs_release_May23DryRun/dev/dev.jsonl")
    save_to_jsonlines(test_entries, "splits/hrs_release_May23DryRun/test/test.jsonl")
    
if __name__ == "__main__":
    main()