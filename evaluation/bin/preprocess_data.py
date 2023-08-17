#!/usr/bin/env python3

import pandas as pd
import numpy as np
from gram2vec import vectorizer


def load_blogs(path:str) -> pd.DataFrame:
    return pd.read_csv(path)

def load_imdb(path:str) -> pd.DataFrame:
    
    df = pd.read_csv(path, sep="\t")
    cols = ["reviewId", "userId", "itemId", "rating", "title", "content"]
    
    # correctly fixtures the column names while preserving all rows
    temp = df.columns
    df.columns = cols
    df.loc[len(df)] = temp
    return df






def main():
    
    IMDB_PATH = "data/imdb/raw/imdb1m-reviews.txt"
    BLOGS_PATH = "data/blogs/raw/blogtext.csv"
    
    
    data = load_imdb(IMDB_PATH)
    
    import ipdb;ipdb.set_trace()
    


if __name__ == "__main__":
    main()
