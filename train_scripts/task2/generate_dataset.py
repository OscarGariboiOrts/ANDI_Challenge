# coding: utf-8

# Author: Òscar Garibo i Orts - Universitat Politècnica de València
# Version: 1
#
# This script stores the .txt file in the directory where the script is run


import datetime
import argparse
import os
import andi
import sys

__version__ = "v1"



def main(args):
    """
        Program's main function.

        Script to generate n trajectories (default = 1.000.000) in 1 dimension under the scope of the ANDI Challenge.

        Params:
         - number: number of trajectories in 1 d to be generated.
         - dimension: dimension of the trajectories (1, 2 or 3)
         - task: task for the data to be used

        Output:
         - task2.txt file with the trajectories
         - ref2.txt files with the labels

    """

    if args.verbose:
        print("[%s]: Starting program. Using version %s of the script" % (datetime.datetime.now(), __version__))

        sys.stdout.flush()

    print("Generating dataset.")

    AD = andi.andi_datasets()
    X1, Y1, X2, Y2, X3, Y3 = AD.andi_dataset(N = int(args.number), tasks = int(args.task), dimensions = int(args.dimension), save_dataset = True)

    print("Program finished.")
    sys.stdout.flush()

    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--number", help='Number of samples to be created', default="10000000")
    parser.add_argument("--dimension", help="Dimension of the trajectories to be generated", default="1")
    parser.add_argument("--task", help="Task where the data is used", default="2")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    main(args)

