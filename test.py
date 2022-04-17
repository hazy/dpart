from dpar import DPAR
from dpar.methods import LogisticRegression, LinearRegression
from hazy_data import datasets


adult_df = datasets["adult"].df
# print(adult_df.head())


model = DPAR(
    methods={
        "age": LinearRegression(epsilon=0.5),
        "sex": LogisticRegression(epsilon=0.5),
    },
    epsilon=1,
)
model.fit(adult_df)


synth_df = model.sample(n_records=adult_df.shape[0],)
print(synth_df.head())
print("BUDGET: ", model.epsilon)
