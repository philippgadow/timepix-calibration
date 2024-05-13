import argparse
from pathlib import Path


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Create a run configuration for peary to take calibration data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--template",
        default="assets/daq_template_h2m.cfg",
        type=Path,
        help="path to input template for run file (device specific)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="path to output file",
    )
    parser.add_argument(
        "--ths_initial",
        type=int,
        default=50,
        help="",
    )
    parser.add_argument(
        "--ths_final",
        type=int,
        default=100,
        help="",
    )
    parser.add_argument(
        "--matrix",
        type=str,
        default="masks/matrix_eq.cfg",
        help="name of matrix file with configuration of individual pixels",
    )
    parser.add_argument(
        "--calibration_folder",
        type=str,
        default="calibration_folder",
        help="name of output directory on caribou board to store runs",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=100,
        help="number of repeated measurements per threshold",
    )
    return parser.parse_args(args)


def main(args=None):
    args = parse_args(args)

    output_file = open(args.output, 'w')

    loop_str = "# set register dac_vthr <value of threshold> <device id>\n"
    loop_str += "# acquire <number of individual frames> <type of data: short0/long1> <filename> <device id>\n"
    for threshold in range(args.ths_initial, args.ths_final, 1):
        loop_str += f"setRegister dac_vthr {threshold} 0\n"
        loop_str += f"acquire {args.repetitions} 0 {args.calibration_folder}/measurement_{threshold} 0\n"
        loop_str += "\n"
        
    with open(args.template) as f:
        output = f.read()

    output = output.format(
        matrix=args.matrix,
        calibration_folder=args.calibration_folder,
        loop_str=loop_str,
    )

    output_file.write(output)
    output_file.close()


if __name__ == "__main__":
    main()
