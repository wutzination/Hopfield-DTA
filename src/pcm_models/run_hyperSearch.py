import pandas as pd
import os
import subprocess


def hyperpara_search():

    grid_df = pd.read_csv("../hyperparaSearch/search_grid.csv")
    print(grid_df)
    for idx, row in grid_df.iterrows():
        file = "hyperparaSearch.py"
        bach_size = "training.batch_size=" + str(int(row["batch_size"]))
        learning_rate = "training.lr=" + str(row["learning_rate"])
        num_hl = "model.architecture.number_hidden_layers=" + str(int(row["num_hl"]))
        num_nodes = "model.architecture.number_hidden_neurons=" + str(int(row["num_nodes"]))
        complete_string = file + " " + bach_size + " " + learning_rate + " " + num_hl + " " + num_nodes

        cmd = "python3 " + complete_string
        os.system(cmd)

if __name__ == "__main__":

    hyperpara_search()