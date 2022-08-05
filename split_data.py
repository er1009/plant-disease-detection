import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-f", "--csv_file", help="Path to the csv data file.", required=True)
    arg("-s", "--save_dir", help="Path to the save dir for the csv's.", required=True)
    return parser.parse_args()


def main(file_path, save_path):
    df = pd.read_csv(file_path)
    sfk = StratifiedKFold(5)
    
    for train_idx, valid_idx in sfk.split(df.index, df['label']):
        df_train = df.iloc[train_idx]
        df_valid = df.iloc[valid_idx]
        break
    
    print(f"train size: {len(df_train)}")
    print(f"valid size: {len(df_valid)}")
    
    df_train.to_csv(save_path + "/train.csv")
    df_valid.to_csv(save_path + "/val.csv")


if __name__ == "__main__":
    args = get_args()
    file_path = args.csv_file
    save_path = args.save_dir
    main(file_path, save_path)
