import numpy as np
import pandas as pd
import json
import os

def analyse_runs():

    #wandb_data_path = '/system/user/user/publicdata/pcm_wurdinger/masterthesis/hyperparaSearch_fnn/wandb/'
    wandb_data_path = '../hyper_data/wandb/'    #local path for tests
    sub_dirs = [x[0] for x in os.walk(wandb_data_path)]

    results_hyper_df = pd.DataFrame()
    index=0
    for it in os.scandir(wandb_data_path):
        if it.is_dir():

            meta_path = it.path + "/files/"+"wandb-metadata.json"
            summary_path = it.path + "/files/" + "wandb-summary.json"
            try:
                with open(summary_path, encoding='utf-8', errors='ignore') as json_data:
                    json_summary = json.load(json_data, strict=False)
                    print(summary_path)
                    data = {"loss_val": json_summary["loss_val"],
                            "ci_val": json_summary["ci_val"],
                            "loss_train": json_summary["loss_train"],
                            "ci_train": json_summary["ci_train"],
                            "dir": it.path
                            }

                    df = pd.DataFrame(data, index=[index])
                    index += 1
                    results_hyper_df = pd.concat([results_hyper_df, df])
            except:
                print(f'Did not work:\n{summary_path}')

    #print(results_hyper_df.nlargest(10, ['ci_val']))
    big_10_df = results_hyper_df.nlargest(10, ['ci_val'])
    print(big_10_df)
    with open('hyper_best_10.txt', 'w') as f:
        for idx, row in big_10_df.iterrows():
            path = row["dir"] + "/files/"+"wandb-metadata.json"
            with open(path, encoding='utf-8', errors='ignore') as json_data:
                json_meta = json.load(json_data, strict=False)
                f.write("ci_val:")
                f.write(str(row["ci_val"]))
                f.write("\t")
                f.write("loss_val:")
                f.write(str(row["loss_val"]))
                f.write("\t")
                f.write(str(json_meta["args"]))
                f.write("\n")


if __name__ == "__main__":
    analyse_runs()
