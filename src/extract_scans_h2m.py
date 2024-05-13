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
    # only use the first 100 rows, as the rest might suffer from uint16 limitations
    data = data[:100]
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
    plotting_args = config.plotting
    base_dir = join("output", "peary", config.name)
    output_dir = join("output", "scans", config.name, measurement_group_name)
    makedirs(output_dir, exist_ok=True)

    plot_dir = join(output_dir, "plots_tot")
    makedirs(plot_dir, exist_ok=True)

    for measurement, column in product(measurements, config.columns):
        data_tag = f'{measurement_group_name}_{measurement}_{column}'
        print(data_tag)
        input_dir = join(base_dir, measurement)

        outputfile_counts_name = join(output_dir, f'ths_counts_{data_tag}.csv')

        h2d = hist.Hist(
            hist.axis.Regular(600, 1100, 1700, name="threshold"),
            hist.axis.Regular(32, -0.5, 31.5, name="values")
        )

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
                
                # fill hist
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                for center, weight in zip(bin_centers, bin_values):
                    h2d.fill(threshold=threshold, values=center, weight=weight)

        # sort outputfile_counts_name by threshold
        header = f'# {config.name}/{measurement} ({column})\n#threshold,counts'
        df = pd.read_csv(outputfile_counts_name, names=['threshold', 'counts'])
        print(df)
        df = df.sort_values(by=['threshold'])
        df.to_csv(outputfile_counts_name, index=False, header=header)

        # # create 2D plots of threshold vs ToT
        # plt.style.use(hep.style.ROOT)
        # fig, ax = plt.subplots()
        # values, _, _ = h2d.to_numpy()
        # plt.title(f"{measurement} ({column})")
        # hep.hist2dplot(h2d, ax=ax, cmap='inferno', vmin=0, vmax=0.1 * np.max(values))
        # fig.tight_layout()
        # fig.savefig(join(plot_dir, f'ths_tot_{data_tag}.png'))
        # plt.close()


if __name__ == "__main__":
    main()
