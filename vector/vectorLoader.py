import numpy as np
import re
from itertools import islice
import os

current_dir = os.path.dirname(os.path.realpath(__file__))


def load_vectors(filename):
    feature_vec = []
    output_vec = []
    try:
        path = ("{}/../resources/{}.txt".format(current_dir, filename))
        # with open(path, "r") as f:
        #     for line in f:
        #     TODO: process the file

        return feature_vec, output_vec

    except IOError as e:
        print("No such file exists", str(e))


