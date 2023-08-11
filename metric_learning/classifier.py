#!/usr/bin/env python3

import numpy as np
import argparse
import pickle
from typing import Tuple
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report


def load_pickle(path:str) -> np.ndarray:
    with open(path, "rb") as reader:
        return pickle.load(reader)
    
def get_pathlib_path(path, filename:str) -> str:
    return list(path.glob(filename))[0]

def get_eval_split_paths(_path:str) -> Tuple[str]:
    path = Path(_path)
    X_eval_path = get_pathlib_path(path, r"X_*.pkl")
    y_eval_path = get_pathlib_path(path, r"y_*.pkl")
    return X_eval_path, y_eval_path

def fit_model(model, X_train:np.ndarray, y_train:np.ndarray):
    return model.fit(X_train, y_train)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path",
                        default="metric_learning/splits/hrs_release_May23DryRun/train")
    parser.add_argument("--eval_set",
                        default="dev")
    args = parser.parse_args()
    
    X_train = load_pickle(args.train_path+"/X_train.pkl")
    y_train = load_pickle(args.train_path+"/y_train.pkl")
    
    eval_path = f"metric_learning/splits/hrs_release_May23DryRun/{args.eval_set}/"
    X_eval_path, y_eval_path = get_eval_split_paths(eval_path)
    X_eval = load_pickle(X_eval_path)
    y_eval = load_pickle(y_eval_path)
    
    fitted_model = fit_model(
        model=LinearSVC(dual="auto"),
        X_train=X_train,
        y_train=y_train
    )

    y_pred = fitted_model.predict(X_eval)
    cl_report = classification_report(y_eval, y_pred)
    print(f"{'~'*50}\nClassification Report:\n{cl_report}\n{'~'*50}")


if __name__ == "__main__":
    main()