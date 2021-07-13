import csv
import argparse


def save_to_csv(path, row):
    with open(path, "a") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def init_csv(path):
    # Current working headers more will be added
    header = ["t model", "t normalization", "t db fetch",
              "t feat reduction", "t search", "t all"]

    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-f", "--file", required=True, help="Path to csv file you want to create"
    )
    args = vars(argParser.parse_args())
    init_csv(args["file"])
