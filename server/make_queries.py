import argparse
import os


from cbir.search import search


def make_queries(file_name):
    result_lines = []
    with open(file_name, "r") as f:
        lines = f.readlines()
        for line in lines:
            query_img_name = line[:-1]
            query_img_path = os.path.join(
                "../../dataset/vacations", query_img_name)

            result = search(query_img_path=query_img_path, cli=True)

            print(query_img_name)

            result_line = " ".join((query_img_name, result))
            print(result_line)
            result_lines.append(result_line)

    with open("results.dat", "w") as f:
        for result_line in result_lines:
            result_line = "".join((result_line, "\n"))
            f.write(result_line)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-f",
        "--file",
        help="Path to file containing query names in columns."
    )
    args = vars(argParser.parse_args())

    make_queries(args["file"])
