import pandas as pd
import numpy as np
import ast


def read_data(fname, sheet_name):
    df = pd.read_excel(fname, sheet_name)
    df["Distance matrix"] = df["Distance matrix"].map(ast.literal_eval).map(np.array)
    df["Permutation Matrix"] = (
        df["Best Permutation Matrix"].map(ast.literal_eval).map(np.array)
    )

    return df[["Distance matrix", "Permutation Matrix"]]


if __name__ == "__main__":
    df = read_data("TSP-4.xlsx", "Training data")
    print(df.head())
