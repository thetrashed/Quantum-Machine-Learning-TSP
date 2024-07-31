import pandas as pd
import numpy as np
import ast


def read_data(fname, sheet_name) -> pd.DataFrame:
    df = pd.read_excel(fname, sheet_name)
    df["Distance Matrix"] = df["Distance Matrix"].map(ast.literal_eval).map(np.array)
    df["Expected Result"] = (
        df["Best Permutation Matrix"].map(ast.literal_eval).map(np.array)
    )

    return df[["Distance Matrix", "Expected Result"]]


if __name__ == "__main__":
    df = read_data("TSP-4.xlsx", "Training data")
    print(df)
