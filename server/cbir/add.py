import argparse
import os

import sys
sys.path.insert(0, "../")
sys.path.insert(0, "./")

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-i", "--images", nargs="+", required=True,
                           help="Pass paths to images you want to add to index.")

    args = vars(argParser.parse_args())
    print(args)
