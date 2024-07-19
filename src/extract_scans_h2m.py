import argparse
import re
import logging
import numpy as np
import hist
import pandas as pd
import shutil

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
    # only use the first 100 rows, as the rest might suffer from uint16 limitations
    data = data[:200]
    n_single_pixel_clusters = data['count'].sum()
    
    # restrict n_bins to 256, no long_count mode for H2M
    n_bins = 256
    bin_edges = np.zeros(n_bins + 1, dtype=float)
    bin_values = np.zeros(n_bins, dtype=float)

    return threshold, n_single_pixel_clusters, bin_edges, bin_values


def main(args=None):
    args = parse_args(args)
    config = Config(args.config_file)
    config.load()

    for measurement_group_name, measurements in config.measurements.items():
        extract_measurement_group(measurement_group_name, measurements, config)


def extract_measurement_group(measurement_group_name, measurements, config):

    input_dir = config.input_dir
    base_dir = join("output", "peary", config.name, config.timestamp)
    makedirs(base_dir, exist_ok=True)

    # copy the directories inside of input_dir to base_dir
    for measurement in measurements:
        input_measurement_dir = join(input_dir, measurement)
        output_measurement_dir = join(base_dir, measurement)

        # only copy if file does not already exist
        try:
            listdir(output_measurement_dir)
        except FileNotFoundError:
            shutil.copytree(input_measurement_dir, output_measurement_dir)

    output_dir = join("output", "scans", config.name, config.timestamp, measurement_group_name)
    makedirs(output_dir, exist_ok=True)

    for measurement, column in product(measurements, config.columns):
        data_tag = f'{measurement_group_name}_{measurement}_{column}'
        print(data_tag)
        input_dir = join(base_dir, measurement)

        outputfile_counts_name = join(output_dir, f'ths_counts_{data_tag}.csv')

        with open(outputfile_counts_name, 'w') as file_count:
            for f in tqdm(sorted(listdir(input_dir))):
                try:
                    threshold, n_single_pixel_cluster, bin_edges, bin_values = process_file(join(input_dir, f))
                except OSError:
                    logging.error('Could not open file ' + f)
                    continue
                except AttributeError:
                    logging.error('Problem opening histogram in file ' + f)
                    continue
                
                # write to file
                line = f"{threshold},{n_single_pixel_cluster}\n"
                file_count.write(line)
                
        # sort outputfile_counts_name by threshold
        header = f'# {config.name}/{config.timestamp}/{measurement} ({column})\n#threshold,counts'
        df = pd.read_csv(outputfile_counts_name, names=['threshold', 'counts'])
        df = df.sort_values(by=['threshold'])
        df.to_csv(outputfile_counts_name, index=False, header=header)


if __name__ == "__main__":
    main()
