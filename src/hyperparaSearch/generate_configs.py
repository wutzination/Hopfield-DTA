import numpy as np
import pandas as pd
import random

if __name__ == "__main__":

    learing_rate = [0.00001, 0.00005, 0.0001]
    batch_size = [512, 1024]
    num_hidden_layers = [4, 5, 6]
    num_nodes = [2048, 4096]

    config_df = pd.DataFrame()
    print(f'Number Configs:\t{len(config_df)}', end="\r")
    while len(config_df) < 50:

        data = {
            "batch_size": random.choice(batch_size),
            "learning_rate": random.choice(learing_rate),
            "num_hl": random.choice(num_hidden_layers),
            "num_nodes": random.choice(num_nodes)
        }
        print(data)
        df = pd.DataFrame(data, index=[0])
        config_df = pd.concat([config_df, df], ignore_index=True)
        config_df.drop_duplicates()
        print(f'Number Configs:\t{len(config_df)}', end="\r")


    config_df.to_csv("search_grid.csv", index=False)
