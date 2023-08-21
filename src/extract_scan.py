import argparse
import re
import logging
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

from os import listdir, makedirs
from os.path import join, basename
from tqdm import tqdm

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
    f = ROOT.TFile.Open(filename)
    h_clusterCharge_1px = f.Get('Clustering4D/detector1/clusterCharge_1px')
    n_single_pixel_clusters = h_clusterCharge_1px.GetEntries()
    f.Close()
    return threshold, n_single_pixel_clusters


def main(args=None):
    args = parse_args(args)
    config = Config(args.config_file)
    config.load()

    base_dir = join("output", "corry", config.name)
    output_dir = join("output", "scans", config.name)
    makedirs(output_dir, exist_ok=True)

    for measurement in config.measurements:
        print(measurement)
        input_dir = join(base_dir, measurement, "output")
        outputfile_name = join(output_dir, f'ths_counts_{measurement}.csv')

        thresholds = []
        n_single_pixel_clusters = []
        
        for f in tqdm(sorted(listdir(input_dir))):
            try:
                threshold, n_single_pixel_cluster = process_file(join(input_dir, f))
            except OSError:
                logging.error('Could not open file ' + f)
                continue
            except AttributeError:
                logging.error('Problem opening histogram in file ' + f)
                continue
            
            thresholds.append(threshold)
            n_single_pixel_clusters.append(n_single_pixel_cluster)
        
        with open(outputfile_name, 'w') as file:
            file.write(f'# {config.name}/{measurement}')
            for ths, counts in zip(thresholds, n_single_pixel_clusters):
                line = f"{ths},{counts}\n"
                file.write(line)


if __name__ == "__main__":
    main()
