import operator
import os
import json
import sys

from collections import defaultdict
from functools import partial
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics
from user_agents import parse

from featurizer import ClickstreamFeaturizer

sys.path.insert(0, r'/Users/veronicahannan/visualization')
from plotter import PythonPlotter


def create_file(input_path):
    my_directory = os.path.dirname(__file__)
    infile_raw = os.path.join(my_directory, input_path)
    infile = os.path.realpath(infile_raw)
    return infile

INFILE = create_file('data/feature_matrix_XXXXXXXX.csv')
PALETTE = 'color_palette.json'

current_directory = create_file('data/')

with open(PALETTE, 'r') as d:
    colors = json.load(d)

plotter = PythonPlotter(colors['purples'], alpha=0.7, fontsize=8, \
                                title_fontsize=14, directory=current_directory)

feature_matrix = pd.read_csv(INFILE, header= 0, delimiter=',')
nfeatures = set(list(feature_matrix.columns.values))
to_remove = filter(lambda x: x.startswith('Unnamed'), nfeatures )
to_remove += set(filter(lambda x: x.endswith('_id'), nfeatures))
nfeatures = list(nfeatures - to_remove)

for parameter in nfeatures:
    for class_label in [1,0]:
        i += 1
        index = i % 2
        temp_data = all_data.loc[all_data['label'] == class_label]
        median = round(temp_data[parameter].median(), 3)
        average = round(temp_data[parameter].mean(), 3)
        stdev = round(temp_data[parameter].std(), 3)
        bin_max = average + stdev * 2
        step = int(median) if int(median) > 1 else 1
        bins = np.arange(0,bin_max,step)
        title = 'Attributable: Median({0}) / Mean({1})'.format(median,average) if \
            class_label == 1 else 'Not Attributable: Median({0}) / Mean({1})'.format(median,average)

        filename = '{0}_class{1}_median.png'.format(parameter, class_label)
        plotter.make_histogram(temp_data[parameter], bins, parameter, 'frequency', \
                                                        title, filename, index*2)