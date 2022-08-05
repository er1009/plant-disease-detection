from glob import glob
import argparse
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-p", "--images_dir", help="Path to the images root dir.", required=True)
    arg("-s", "--save_dir", help="Path to the save dir for the data csv.", required=True)
    return parser.parse_args()


def main(images_path, save_path):
    image_paths = []
    labels = []
    image_id = []
    for dir in glob(images_path + '/*'):
        label = dir.split('/')[-1]
        for path in glob(dir +'/*'):
            image_paths.append(path)
            labels.append(label)
            image_id.append(path.split("/")[-1].split(".")[0])
    
    dict_ = {"image_path":image_paths, "image_id":image_id, "label":labels}
    df = pd.DataFrame(data=dict_)
    df.to_csv(save_path + "/train.csv")


if __name__ == "__main__":
    args = get_args()
    images_path = args.images_dir
    save_path = args.save_dir
    main(images_path, save_path)
