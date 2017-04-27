# Utilities for saving and reloading results

from pickle import load, dump


def save_model(model, name):
    with open("{}.pkl", "w") as f:
        dump(model, f)


def load_model(filename):
    with open(filename, "r") as f:
        model = load(f)
    return model
