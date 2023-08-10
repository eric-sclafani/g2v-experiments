#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import time
import pickle
from pathlib import Path
from typing import Iterable, List, Tuple
from dataclasses import dataclass
from more_itertools import distinct_combinations

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

@dataclass
class Document:
    
    entry:pd.Series

    @property
    def vector(self) -> np.ndarray:
        """Extracts only numeric values from a series"""
        return self.entry.apply(pd.to_numeric, errors='coerce').dropna().values
    
    @property
    def author_id(self) -> str:
        return self.entry["authorIDs"]
        
    @property
    def document_id(self)-> str:
        return self.entry.name
    
@dataclass
class DocumentPair:
    
    doc1: Document
    doc2: Document
    
    @property
    def doc_id_pair(self) -> Tuple[str,str]:
        return self.doc1.document_id, self.doc2.document_id
    
    def has_different_author_id(self):
        return self.doc1.author_id != self.doc2.author_id
    
    
    @classmethod
    def from_dataframe(cls, df:pd.DataFrame):
        d1 = Document(df.iloc[0])
        d2 = Document(df.iloc[1])
        return cls(d1, d2)

@measure_time
def apply_vectorizer(path:str) -> pd.DataFrame:
    df = vectorizer.from_jsonlines(path)
    df["authorIDs"] = df["authorIDs"].apply(lambda x: "".join(x))
    return df
    
def get_unique_author_ids(df:pd.DataFrame) -> List[str]:
    return df["authorIDs"].unique().tolist()

def get_author_doc_vectors(df:pd.DataFrame, author_id:str) -> pd.DataFrame:
    return df.loc[df["authorIDs"] == author_id]

def to_array(iter:List[float]) -> np.ndarray:
    return np.array(iter)

def get_string(series:pd.Series) -> str:
    return series.fullText.values[0]

def get_author(series:pd.Series) -> str:
    return series.authorIDs.values[0]

def difference(pair:Tuple[List, List]) -> np.ndarray:
    """Calculates the element-wise difference for two vectors"""
    return np.abs(to_array(pair[0]) - to_array(pair[1]))

def calculate_difference_from_iterable(pairs:Iterable[Tuple]) -> np.ndarray:
    """Calculates the element-wise difference for a collection of vector pairs"""
    return np.array([difference(pair) for pair in pairs])

@measure_time
def create_same_author_similarity_vectors(author_ids:List[str], data:pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each author, creates all possible distinct combinations of vector pairs and calculate their element-wise similarity
    """
    X_train = []
    y_train = []
    for author_id in author_ids:
        vectors = get_author_doc_vectors(data, author_id).select_dtypes(include=np.number).values
        same_author_vector_pairs = distinct_combinations(vectors.tolist(), r=2)
        similarity_vectors = 1 - calculate_difference_from_iterable(same_author_vector_pairs)
        
        for vector in similarity_vectors:
            X_train.append(vector)
            y_train.append(1)
            
    return np.array(X_train), np.array(y_train)

@measure_time
def sample_n_pairs(data:pd.DataFrame, n:int) -> List[DocumentPair]:
    """Creates n amount of different author document pairs"""
    
    seen_doc_id_pairs = [] # ensure no two doc pairs have the same exact documents
    pairs = []
    while len(pairs) != n:
        sampled = data.sample(n=2)
        pair = DocumentPair.from_dataframe(sampled)
        if pair.has_different_author_id() and pair.doc_id_pair not in seen_doc_id_pairs:
            pairs.append(pair)
            seen_doc_id_pairs.append(pair.doc_id_pair)
    return pairs

@measure_time 
def create_different_author_similarity_vectors(pairs:List[DocumentPair]) -> Tuple[np.ndarray, np.ndarray]:
    """Creates similarity vectors using documents from different authors. The amount is equal to the # of same author vectors"""
    X_train = []
    y_train = []
    for pair in pairs:
        v1 = pair.doc1.vector
        v2 = pair.doc2.vector
        similarity_vector = 1 - difference((v1, v2))
        X_train.append(similarity_vector)
        y_train.append(0)
        
    return np.array(X_train), np.array(y_train)

def write_to_file(obj, path:str):
    with open(path, "wb") as writer:
        pickle.dump(obj, writer)
        
def generate_ml_data(path:str):
    """Applies metric learning data generation steps to a given dataset"""
    print("Vectorizing data...\n")
    document_vectors = apply_vectorizer(path)    
    author_ids = get_unique_author_ids(document_vectors)
    
    print("Creating same author similarity vectors...\n")
    same_author_vectors, same_author_labels = create_same_author_similarity_vectors(author_ids, document_vectors)
    
    print(f"Creating different author document pairs...\n")
    n = same_author_vectors.shape[0]
    pairs = sample_n_pairs(document_vectors, n)
 
    print("Creating different author similarity vectors...\n")
    diff_author_vectors, diff_author_labels = create_different_author_similarity_vectors(pairs)
    
    X_train = np.concatenate([same_author_vectors, diff_author_vectors], axis=0)
    y_train = np.concatenate([same_author_labels,diff_author_labels])
    
    return X_train, y_train


@measure_time
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--dir",
                        help="Directory containing the train, dev, and test directories with JSONL files",
                        default="metric_learning/splits/hrs_release_May23DryRun")
    
    args = parser.parse_args()
    data_dir = Path(args.dir)

    for subdir in data_dir.iterdir():
        
        if subdir.name == "train":
            in_path = f"{args.dir}/train/train.jsonl"
            X_out_path = f"{args.dir}/train/X_train.pkl"
            y_out_path = f"{args.dir}/train/y_train.pkl"
        elif subdir.name == "dev":
            in_path = f"{args.dir}/dev/dev.jsonl"
            X_out_path = f"{args.dir}/dev/X_dev.pkl"
            y_out_path = f"{args.dir}/dev/y_dev.pkl"
        elif subdir.name == "test":
            in_path = f"{args.dir}/test/test.jsonl"
            X_out_path = f"{args.dir}/test/X_test.pkl"
            y_out_path = f"{args.dir}/test/y_test.pkl"
        
        print(f"Currently processing: '{in_path}'")
        X, y = generate_ml_data(in_path)
        
        assert X.shape[0] == y.shape[0]
        write_to_file(X, X_out_path)
        write_to_file(y, y_out_path)
    
    
if __name__ == "__main__":
    main()
