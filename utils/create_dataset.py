import argparse
import os
import random
import shutil

parser = argparse.ArgumentParser(
    description="Copy N images from one directory to another"
)

parser.add_argument(
    "-d",
    "--dest_folder",
    metavar="d",
    type=str,
    help="Folder where you want files to be copied to.",
)

parser.add_argument(
    "-s",
   "--src_folder",
    metavar="s",
    type=str,
    help="Folder from where you want to copy files.",
)


parser.add_argument(
    "-n",
    "--n_elements",
    metavar="n",
    type=int,
    help="Number of elements to copy.",
)

args = parser.parse_args()

src_dir = args.src_folder

for directory in os.listdir(src_dir):
    # print(os.path.join(args.src_folder, directory))
    curr_dir = os.path.join(src_dir, directory)
    files = [file for file in os.listdir(curr_dir)]
    for x in range(len(files)):
        copy_file = files[x]
        print(copy_file)
        shutil.copyfile(
            os.path.join(curr_dir, copy_file), os.path.join(
                args.dest_folder, copy_file)
        )
