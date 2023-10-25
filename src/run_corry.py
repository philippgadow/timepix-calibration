import argparse
import logging
from os import makedirs, listdir
from os.path import join, basename, splitext, abspath, exists
from re import search
from shutil import copyfile
from subprocess import run
from tqdm import tqdm

from config import Config


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Process measurements with corry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "config_file",
        help="path to config file",
    )

    return parser.parse_args(args)


def assemble_config_files(name, matrix_file, detectors_file, config_template):
    base_dir = join("output", "corry", name)
    config_dir = join(base_dir, "configs")
    output_directory = abspath(join(base_dir, "output"))
    makedirs(config_dir, exist_ok=True)
    makedirs(output_directory, exist_ok=True)
    for f in [matrix_file, detectors_file]:
        copyfile(f, join(config_dir, basename(f)))
    
    config_file = join(config_dir, "run.conf")
    detectors_file_local =join(config_dir, basename(detectors_file))
    input_directory = join(base_dir, "input", "@RunNumber@")
    histogram_file = "histograms_@RunNumber@.root"

    with open(config_template) as config_in, open(config_file, 'w') as config_out:
        for line in config_in.readlines():
            if line.startswith('input_directory ='): config_out.write(f'input_directory = "{input_directory}"\n')
            elif line.startswith('detectors_file ='): config_out.write(f'detectors_file = "{detectors_file_local}"\n')
            elif line.startswith('output_directory ='): config_out.write(f'output_directory = "{output_directory}"\n')
            elif line.startswith('histogram_file ='): config_out.write(f'histogram_file = "{histogram_file}"\n')
            else: config_out.write(line)

    return base_dir


def main(args=None):
    args = parse_args(args)
    config = Config(args.config_file)
    config.load()

    htcondor_dir = join("output", "batch")
    makedirs(htcondor_dir, exist_ok=True)

    # unpack measurements from the config file groups and process all of them
    measurements = sum(config.measurements.values(),[])
    for measurement in measurements:
        print(measurement)
        input_dir = join(config.input_dir, measurement)
        name = join(config.name, measurement)
        base_dir = assemble_config_files(name, config.matrix_file, config.detectors_file, config.config_template)

        thresholds = []
        # list content of input directory: contains measurements to be processed
        for input_file in tqdm(sorted(listdir(input_dir))):
            measurement_name, file_ext = splitext(input_file)
            if file_ext != '.csv': continue
            # extract threshold from file name
            try:
                threshold = int(search(r"\d+", measurement_name).group())
            except AttributeError:
                logging.error(f'Skipping file {input_file}')
                continue
            thresholds.append(threshold)

            # prepare directory structure, corry requires input data to be in directory
            data_dir = join(base_dir, "input", f"{threshold}")
            makedirs(data_dir, exist_ok=True)
            if not exists(join(data_dir, input_file)):
                copyfile(join(input_dir, input_file), join(data_dir, input_file))
                copyfile(config.matrix_file, join(data_dir, basename(config.matrix_file)))

        # submit jobs
        jobsub_executable = "./corryvreckan/jobsub/jobsub.py"
        config_file = join(base_dir, "configs", "run.conf")

        # local test
        # run([jobsub_executable, "-c", config_file, "1300"])

        # local submission
        for t in thresholds:
            output_file = join(base_dir, "output", f"histograms_{t}.root")
            if exists(output_file): continue
            run([jobsub_executable, "-c", config_file, f"{t}"])


        # HTCondor submission
        # for t in thresholds:
        #     output_file = join(base_dir, "output", f"histograms_{t}.root")
        #     if exists(output_file): continue
        #     if config.htcondor_config:
        #         run([jobsub_executable, "-c", config_file, "--htcondor-file", config.htcondor_config, f"{t}"])
        #     else:
        #         run([jobsub_executable, "-c", config_file, f"{t}"])


if __name__ == "__main__":
    main()
