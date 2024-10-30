# Pixel calibration

> This software provides means of calibrating silicon pixel detectors. 
> It uses input from `csv` files which can be either read in stand-alone or using the [corrywreckan](https://gitlab.cern.ch/corryvreckan/corryvreckan) software for the efficient analysis of measurements.

## Setup

Please follow these instructions to install the software. Doing so will download the software and the corrywreckan project as a git submodule. If you forgot to add the `--recursive` argument for `git clone`, you can enter `git submodule init && git submodule update` to obtain corrywreckan.

It is assumed that you are running this software on lxplus. Otherwise, some manual intervention might be needed to install corrywreckan.


```bash
# obtain project from github
git clone --recursive git@github.com:philippgadow/pixel-calibration.git 
cd pixel-calibration

# execute setup script
source setup.sh
```

This software will be installed in a virtual environment.


## Use without corrywreckan (for H2M)

The calibration involves two steps and assumes you have acquired data previously in either `root` format or `csv` format, with one measurement file per threshold, all stored in one directory.

1. Extracting data from the `root` or `csv` files
2. Analysis of data to obtain calibration

The design paradigm of this software is to disentangle configuration and code, therefore all information relevant for a device calibration has to be provided in a configuration file in `yaml` format.

### Preparation 

For a calibration of a device, you should provide the following:

- Configuration file: look at [`data/assets/H2M-2_1p2V_ikrum10/calibration_config.yaml`](https://github.com/philippgadow/pixel-calibration/blob/main/data/assets/H2M-2_1p2V_ikrum10/calibration_config.yaml) for an example

### Running

You can run the full calibration with the following commands.

```bash
python src/extract_scans.py data/assets/H2M-2_1p2V_ikrum10/calibration_config.yaml
```

This will extract the s-curves from the input files.

You find the output in `output/`.


```bash
python src/process_scans.py data/assets/H2M-2_1p2V_ikrum10/calibration_config.yaml
```

This will perform the actual calibration. You find the output in `output/`.


## Use with corrywreckan (for CLICpix2)

The calibration involves three steps and assumes you have acquired data previously in a format accessible to [corrywreckan](https://gitlab.cern.ch/corryvreckan/corryvreckan) `EventLoader` class appropriate to the device under test. 

1. Running corrywreckan
2. Extracting data from corrywreckan output `root` files
3. Analysis of data to obtain calibration

The design paradigm of this software is to disentangle configuration and code, therefore all information relevant for a device calibration has to be provided in a configuration file in `yaml` format.

### Preparation 

For a calibration of a device, you should provide the following:

- Configuration file: look at [`data/assets/CLICpix2-1185_1_E1/calibration_config.yaml`](https://github.com/philippgadow/pixel-calibration/blob/main/data/assets/CLICpix2-1185_1_E1/calibration_config.yaml) for an example
- Measurement data: store all `csv` or `root` files in a single directory and specify it in `input_dir:` in your config file
- Matrix file: for device encoding trimming and masking of individual pixels, specified in `matrix_file:` in your config file
- Corrywreckan instructions: for processing of data using corrywreckan a dummy geometry file is needed and a file with corrywreckan configuration. Both is provided in the directory [`data/corry`](https://github.com/philippgadow/pixel-calibration/tree/main/data/corry)

### Running

You can run the full calibration with the following commands.

```bash
python src/run_corry.py data/assets/CLICpix2-1185_1_E1/calibration_config.yaml
```

This will assemble your data and launch corrywreckan jobs. You might want to modify the content of `src/run_corry.py`, e.g. to use batch submission or to change the `copy` command of data to a `move` command if your inputs are very large. One job per threshold of the threshold scan is scheduled and you will get exactly one output file.

You find the output in `output/corry`.

```bash
python src/extract_scans.py data/assets/CLICpix2-1185_1_E1/calibration_config.yaml
```

This will extract the s-curves from the corrywreckan outputs.

You find the output in `output/scans`.


```bash
python src/process_scans.py data/assets/CLICpix2-1185_1_E1/calibration_config.yaml
```

This will perform the actual calibration. You find the output in `output/scans`.
