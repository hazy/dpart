import pickle
import pandas as pd


# data path, columns and dtypes
path = "adult.data"
test_path = "adult.test"

dtypes = {
    "age": "int",
    "workclass": "category",
    "fnlwgt": "int",
    "education": "category",
    "education_num": "category",
    "marital_status": "category",
    "occupation": "category",
    "relationship": "category",
    "race": "category",
    "sex": "category",
    "capital_gain": "int",
    "capital_loss": "int",
    "hours_per_week": "int",
    "native_country": "category",
    "income": "category",
}
columns = list(dtypes.keys())

tiny_columns = [
    "age",
    "education",
    "marital_status",
    "relationship",
    "sex",
    "income",
]
tiny_dtypes = {k: v for k, v in dtypes.items() if k in tiny_columns}


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
                 names=columns,
                 dtype=dtypes,
                 engine="python")
test_df = pd.read_csv(test_path,
                      sep=r",\s+",
                      names=columns,
                      dtype=dtypes,
                      engine="python",
                      skiprows=1)


# clean test labels
test_df["income"] = test_df["income"].replace("<=50K.", "<=50K").replace(">50K.", ">50K")
# clean dtypes
test_df["native_country"] = test_df["native_country"].cat.set_categories(df["native_country"].cat.categories)


# save adult
df.to_pickle("adult.pkl.gz", compression="gzip")
test_df.to_pickle("adult_test.pkl.gz", compression="gzip")


# get tiny adult
tiny_df = df[tiny_columns]
tiny_test_df = test_df[tiny_columns]

# get tiny adult bounds
tiny_df_bounds = get_bounds(tiny_df)

# save tiny adult
tiny_df.to_pickle("tiny_adult.pkl.gz", compression="gzip")
tiny_test_df.to_pickle("tiny_adult_test.pkl.gz", compression="gzip")
with open("tiny_adult_bounds.pkl", "wb") as f:
    pickle.dump(tiny_df_bounds, f, pickle.HIGHEST_PROTOCOL)
