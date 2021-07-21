import argparse
import os


from cbir import search


def make_queries(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
        print(lines)
        for line in lines:
            query_img_name = line[:-1]
            query_img_path = os.path.join(
                "../../dataset/vacations", query_img_name)

            print(query_img_path)
            # search(query_img_name, cli=True, dataset="")


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-f",
        "--file",
        help="Path to file containing query names in columns."
    )
    args = vars(argParser.parse_args())

    make_queries(args["file"])
