import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("-q", "--query", required=True, help="Path to the query image")
argParser.add_argument(
    "-r", "--result_path", required=True, help="Path to results directory"
)
args = vars(argParser.parse_args())
