import pickle
import pandas as pd
from tqdm import tqdm
from time import time
from tempfile import TemporaryDirectory
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator

from dpart.engines import PrivBayes


def get_data():
    train_df = pd.read_pickle("data/adult/adult.pkl.gz")
    bounds = None
    return train_df, bounds


def DS_baseline(train_df, bounds, epsilon):
    threshold_value = 20

    # specify categorical attributes
    categorical_attributes = {col: True for col, series in train_df.items() if series.dtype.name == "category"}
    # specify which attributes are candidate keys of input dataset.

    # A parameter in Differential Privacy. It roughly means that removing a row in the input dataset will not
    # change the probability of getting the same output more than a multiplicative difference of exp(epsilon).
    # Increase epsilon value to reduce the injected noises. Set epsilon=0 to turn off differential privacy.

    # The maximum number of parents in Bayesian network, i.e., the maximum number of incoming edges.
    degree_of_bayesian_network = 2

    with TemporaryDirectory() as temp_dir:
        input_file = f"{temp_dir}/input.csv"
        model_file = f"{temp_dir}/model.json"

        train_df.to_csv(input_file, index=False)
        start_time = time()
        describer = DataDescriber(category_threshold=threshold_value)
        describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_file,
                                                                epsilon=epsilon,
                                                                k=degree_of_bayesian_network,
                                                                attribute_to_is_categorical=categorical_attributes)

        describer.save_dataset_description_to_file(model_file)
        train_time = time()
        generator = DataGenerator()
        _ = generator.generate_dataset_in_correlated_attribute_mode(train_df.shape[0], model_file)
        gen_time = time()

    return train_time - start_time, gen_time - train_time


def DPART_baseline(train_df, bounds, epsilon):
    start_time = time()
    dpart = PrivBayes(n_parents=2, epsilon=epsilon, bounds=bounds)
    dpart.fit(train_df)
    train_time = time()
    _ = dpart.sample(train_df.shape[0])
    gen_time = time()
    return train_time - start_time, gen_time - train_time


if __name__ == "__main__":
    train_df, bounds = get_data()
    n_exp = 5
    to_test = {"DataSynthesizer": DS_baseline, "DPART": DPART_baseline}
    results = []
    for idx in tqdm(range(n_exp)):
        for b_name, baseline in tqdm(to_test.items(), leave=False):
            train_time, gen_time = baseline(train_df=train_df, bounds=bounds, epsilon=1)
            results.append({"baseline": b_name, "idx": idx, "train": train_time, "gen": gen_time})

    pd.DataFrame(results).to_csv("data/speed_results.csv", index=False)
