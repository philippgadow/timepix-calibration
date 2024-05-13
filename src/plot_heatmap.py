import argparse
import re
import logging
import numpy as np
import hist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mplhep as hep
import pandas as pd

from itertools import product
from os import listdir, makedirs
from os.path import join, basename
from tqdm import tqdm
from json import dump

from config import Config


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Extract single pixel clusters from corry analysis for different thresholds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "config_file",
        help="path to config file",
    )

    return parser.parse_args(args)


def process_file(filename):
    # get threshold from filename
    threshold = re.search(r'\d+', basename(filename)).group()

    # get number of single pixel hits from corry clustering results
    # open csv file with pandas, ignore first 7 rows, ignore comments starting with "====", split by ","
    # header is (row, column, acq_mode, value)
    data = pd.read_csv(filename, skiprows=7, comment='=', sep=',', names=['row', 'column', 'acq_mode', 'count'])


    return threshold, data


def main(args=None):
    args = parse_args(args)
    config = Config(args.config_file)
    config.load()

    for measurement_group_name, measurements in config.measurements.items():
        extract_measurement_group(measurement_group_name, measurements, config)


def extract_measurement_group(measurement_group_name, measurements, config):
    plotting_args = config.plotting
    base_dir = join("output", "peary", config.name)
    output_dir = join("output", "scans", config.name, measurement_group_name)
    makedirs(output_dir, exist_ok=True)

    plot_dir = join(output_dir, "plots_heatmaps")
    makedirs(plot_dir, exist_ok=True)

    for measurement, column in product(measurements, config.columns):
        data_tag = f'{measurement_group_name}_{measurement}_{column}'
        print(data_tag)
        input_dir = join(base_dir, measurement)

        for f in tqdm(sorted(listdir(input_dir))):
            try:
                threshold, data = process_file(join(input_dir, f))
            except OSError:
                logging.error('Could not open file ' + f)
                continue
            except AttributeError:
                logging.error('Problem opening histogram in file ' + f)
                continue
            
            # print(threshold)
            # print(data)


            # create 2D histogram with rows and columns as axes and values as counts
            h2d = hist.Hist(
                hist.axis.Regular(64, -0.5, 63.5, name="col"),
                hist.axis.Regular(16, -0.5, 15.5, name="row"),
            )

            # loop over data and fill histogram
            for column, row, _, value in data.itertuples(index=False):
                h2d.fill(column, row, weight=value)
                
            # create 2D plots of histogram
            plt.style.use(hep.style.ROOT)
            fig, ax = plt.subplots()
            hep.hist2dplot(h2d, ax=ax, cmap='inferno')
            fig.tight_layout()
            fig.savefig(join(plot_dir, f'heatmap_{data_tag}_{threshold}.png'))
            plt.close()


if __name__ == "__main__":
    main()
