import os
# import glob
import argparse

import numpy as np
import pandas as pd

from data.data_layout import CITY_NAMES, CITY_TRAIN_ONLY
from metrics.get_scores import get_score_names


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parser for CLI arguments to run model.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--test_pred_path", type=str, default=None, required=True,
                        help="Test pred path.")
    parser.add_argument("--create_table", type=str, default="True", required=False, choices=["True", "False"],
                        help="'Boolean' specifying if get_scores_to_table function should be called.")
    parser.add_argument("--agg_table", type=str, default="False", required=False, choices=["True", "False"],
                        help="'Boolean' specifying if aggregate_tables function should be called.")
    parser.add_argument("--agg_nr", type=int, default=None, required=False,
                        help="Nr. of tables to aggregate, used to cycle through folders.")

    return parser


def get_scores_to_table(test_pred_path: str):

    scorenames = get_score_names()
    colnames = list(np.delete(scorenames[1:-1].split(", "), [4,5,6,7]))
    colnames.insert(0, "uq_method")
    colnames.insert(0, "city")
    
    cities = [city for city in CITY_NAMES if city not in CITY_TRAIN_ONLY]
    # limit to actually available cities as determined by folder structure
    # cities = cities[:len(glob.glob(f"{test_pred_path}/scores/*", recursive=True))]
    
    mask = ["", "_mask"]
    channels = ["speed", "vol"]
    uq_methods = ["point", "combo", "ensemble", "bnorm", "tta", "patches"]
    
    for m in mask:
        for ch in channels:
    
            df_list_of_lists = []
    
            for city in cities:
                for uq in uq_methods:
    
                    filename = f"scores_{uq}_{ch}{m}.txt"
                    try:
                        scores = list(np.loadtxt(os.path.join(test_pred_path, "scores", city, filename)))
                    except OSError:
                        scores = ["N/A" for i in range(14)]

                    scores_df = []
                    scores_df.append(city.lower())
                    scores_df.append(uq)
                    scores_df.append(str(scores[0]) + " +/- " + str(scores[4]))
                    scores_df.append(str(scores[1]) + " +/- " + str(scores[5]))
                    scores_df.append(str(scores[2]) + " +/- " + str(scores[6]))
                    scores_df.append(str(scores[3]) + " +/- " + str(scores[7]))
                    scores_df.append(str(scores[8]))
                    scores_df.append(str(scores[9]))
                    scores_df.append(str(scores[10]))
                    scores_df.append(str(scores[11]))
                    scores_df.append(str(scores[12]))
                    scores_df.append(str(scores[13]))

                    df_list_of_lists.append(scores_df)

            df = pd.DataFrame(df_list_of_lists, columns=colnames)
            dfname = f"results_{ch}{m}.csv"
            df.to_csv(os.path.join(test_pred_path, "scores", dfname), index=False, na_rep='N/A')
    
    # path = sorted(glob.glob(f"{test_pred_path}/scores/{city}/scores_{uq_method}_*.txt", recursive=True))
    # s = path[0]
    # s.split("/")[-1][:-4].split("_")


def aggregate_tables(test_pred_path: str, agg_nr: int):

    scorenames = get_score_names()
    colnames = list(np.delete(scorenames[1:-1].split(", "), [4,5,6,7]))
    colnames.insert(0, "uq_method")
    colnames.insert(0, "city")

    fnames = ["results_speed_mask", "results_speed", "results_vol_mask", "results_vol"]
    folder = test_pred_path.split("/")[-2]

    for f in fnames:
        df = pd.DataFrame(columns=colnames)

        for nr in range(1, agg_nr+1):
            path = os.path.join(os.path.split(test_pred_path)[0], f"{folder}_{nr}", "scores", f + ".csv")
            df = pd.concat([df, pd.read_csv(path)])

        # split str columns and make numeric for means
        df[["m_gt", "rm", "std_gt"]] = df["mean_gt"].str.split(" ", -1, expand=True)
        df[["m_pred", "rm", "std_pred"]] = df["mean_pred"].str.split(" ", -1, expand=True)
        df[["m_unc", "rm", "std_unc"]] = df["mean_unc"].str.split(" ", -1, expand=True)
        df[["m_mse", "rm", "std_mse"]] = df["mean_mse"].str.split(" ", -1, expand=True)
        df = df.drop(["rm", "mean_gt", "mean_pred", "mean_unc", "mean_mse"], axis=1)
        df.iloc[:, 2:] = df.iloc[:, 2:].apply(pd.to_numeric)

        # group by (city, uq_method) and take means over all columns
        res = df.groupby(["city", "uq_method"]).mean().reset_index()
        # res = res.round({"PI_width": 3, "cover": 3, "ENCE": 3, "CoV": 3,
        #                  "corr": 3, "sp_corr": 3, "m_gt": 2, "std_gt": 2,
        #                  "m_pred": 2, "std_pred": 2, "m_unc": 2, "std_unc": 2,
        #                  "m_mse": 2, "std_mse": 2})
        res = res.round(3)

        # sort uq methods by desired order
        sorter = ["point", "combo", "ensemble", "bnorm", "tta", "patches"]
        res["uq_method"] = res["uq_method"].astype("category").cat.set_categories(sorter)
        res = res.sort_values(["city", "uq_method"]).groupby("city").head(6)

        # create new str columns
        s = " +- "
        res.insert(2, "mean_gt", res["m_gt"].astype(str) + s + res["std_gt"].astype(str))
        res.insert(3, "mean_pred", res["m_pred"].astype(str) + s + res["std_pred"].astype(str))
        res.insert(4, "mean_unc", res["m_unc"].astype(str) + s + res["std_unc"].astype(str))
        res.insert(5, "mean_mse", res["m_mse"].astype(str) + s + res["std_mse"].astype(str))
        res = res.drop(["m_gt", "std_gt", "m_pred", "std_pred", "m_unc", "std_unc", "m_mse", "std_mse"], axis=1)

        dfname = f"{f}_mean_{agg_nr}.csv"
        res.to_csv(os.path.join(os.path.split(test_pred_path)[0], dfname), index=False, na_rep='N/A')


def main():
    parser = create_parser()
    args = parser.parse_args()
    if eval(args.create_table) is not False:
        get_scores_to_table(args.test_pred_path)
    if eval(args.agg_table) is not False:
        aggregate_tables(args.test_pred_path, args.agg_nr)


if __name__ == "__main__":
    main()
