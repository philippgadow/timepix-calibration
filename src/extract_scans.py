import argparse
from os import makedirs
from os.path import join
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


def main(args=None):
    args = parse_args(args)
    config = Config(args.config_file)
    config.load()

    for measurement_group_name, measurements in config.measurements.items():
        extract_measurement_group(measurement_group_name, measurements, config)


def extract_measurement_group(measurement_group_name, measurements, config):
    base_dir = join("output", config.name, config.timestamp)
    makedirs(base_dir, exist_ok=True)

    for measurement in measurements:
        measurement.extract_measurement(base_dir)


if __name__ == "__main__":
    main()
