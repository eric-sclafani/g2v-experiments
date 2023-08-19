#!/usr/bin/env python3

import pandas as pd
import polars as pl
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

def load_all_the_news(path:str) -> pl.DataFrame:
    return pl.read_csv(path)









def main():
    
    IMDB_PATH = "data/imdb/raw/imdb1m-reviews.txt"
    BLOGS_PATH = "data/blogs/raw/blogtext.csv"
    ALL_THE_NEWS_PATH = "evaluation/data/all-the-news/raw/all-the-news-2-1.csv.gz"
    
    
    imdb = load_imdb(IMDB_PATH)
    
    import ipdb;ipdb.set_trace()
    


if __name__ == "__main__":
    main()
