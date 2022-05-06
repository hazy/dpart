import pickle
import pandas as pd


# data path, columns and dtypes
path = "adult.data"
test_path = "adult.test"

dtypes = {
    "age": "int",
    "education": "category",
    "marital_status": "category",
    "relationship": "category",
    "sex": "category",
    "income": "category",
}
columns = list(dtypes.keys())
all_columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


def get_bounds(df):
    bounds = {}
    for col in df:
        col_data = df[col]
        if col_data.dtype.kind in "fui":
            bounds[col] = {"min": col_data.min(),
                           "max": col_data.max()}
        if col_data.dtype.name == "category":
            bounds[col] = {"categories": col_data.unique().to_list()}

    return bounds


# load data
df = pd.read_csv(path,
                 sep=r",\s+",
                 names=all_columns,
                 usecols=columns,
                 dtype=dtypes,
                 engine="python")
test_df = pd.read_csv(test_path,
                      sep=r",\s+",
                      names=all_columns,
                      usecols=columns,
                      dtype=dtypes,
                      engine="python",
                      skiprows=1)


# clean test labels
test_df["income"] = test_df["income"].replace("<=50K.", "<=50K").replace(">50K.", ">50K")


# get df bounds
df_bounds = get_bounds(df)


# save
df.to_pickle("tiny_adult.pkl.gz", compression="gzip")
test_df.to_pickle("tiny_adult_test.pkl.gz", compression="gzip")
with open("tiny_adult_bounds.pkl", "wb") as f:
    pickle.dump(df_bounds, f, pickle.HIGHEST_PROTOCOL)
