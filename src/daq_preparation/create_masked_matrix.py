import argparse
import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Create a matrix with masking (deactivating pixels) from a csv file applied",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--template",
        required=True,
        type=Path,
        help="path to input matrix to which masking will be applied",
    )

    parser.add_argument(
        "--mask",
        required=True,
        type=Path,
        help="path to csv file with masking information",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="path to output file",
    )
    return parser.parse_args(args)


def read_mask_file(mask_file):
    columns = ['col', 'row']
    dtype = {
        'col': np.int32,
        'row': np.int32,
    }
    return pd.read_csv(mask_file, comment='#', names=columns, dtype=dtype)
     

def process_line(line, df_mask):
    # ignore commented lines
    if line.startswith("#"): return line
    # line structure
    # ROW COL mask threshold countingmode testpulse longcnt

    # TODO: replace by regular expression
    l = line.split()
    assert(len(l) == 7)
    row, col, mask, threshold, countingmode, testpulse, longcnt = int(l[0]), int(l[1]), int(l[2]), int(l[3]), int(l[4]), int(l[5]), int(l[6])

    # check if entry should be masked
    if (df_mask.loc[(df_mask['col'] == col) & (df_mask['row'] == row)]).any().all():
        mask = 1
        return f"{row} {col} {mask} {threshold} {countingmode} {testpulse} {longcnt}\n"
    return line


def main(args=None):
    args = parse_args(args)

    output = open(args.output, 'w')

    # read mask
    df_mask = read_mask_file(args.mask)

    with open(args.template) as f:
        for line in tqdm(f.readlines()):
            out_line = process_line(line, df_mask)
            output.write(out_line)
    output.close()

if __name__ == "__main__":
    main()
