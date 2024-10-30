import argparse
from os.path import join
from config import Config
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Plot threshold scans.",
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
    
    # process measurement groups
    for measurement_group_name, measurements in config.measurements.items():
        process_measurement_group(measurement_group_name, measurements, config)


def process_measurement_group(measurement_group_name, measurements, config):
    plotting_args = config.plotting
    base_dir = join("output", config.name, config.timestamp)

    # process measurements
    data = []
    for measurement in measurements:
        fit_results = measurement.collect_data(base_dir, plotting_args)
        energy_keV = float(fit_results['energy'])
        energy_eh = energy_keV / 3.65 * 1000. # 3.65 eV to create e/h pair in Si
        row = [energy_keV, energy_eh, float(fit_results['threshold']), float(fit_results['threshold_width'])]
        data.append(row)

    header = ['energy_keV', 'energy_eh', 'threshold', 'threshold_err']
    data = np.array(data)
    np.savetxt(join(base_dir, 'fitresults.csv'), data, delimiter=',', header=','.join(header), comments='', fmt='%s')
    # data = np.genfromtxt(join(base_dir, 'fitresults.csv'), delimiter=',', skip_header=1)

    x = data[:,1]
    y = data[:,2]
    y_err = data[:,3]

    weights_polyfit = np.power(np.array(y_err), -1)

    try:
        coefficients, cov_matrix = np.polyfit(x, y, deg=1, w=weights_polyfit, cov=True)
        slope = coefficients[0]
        intercept = coefficients[1]

        slope_uncertainty = np.sqrt(cov_matrix[0, 0])
        intercept_uncertainty = np.sqrt(cov_matrix[1, 1])
        inverse_slope_uncertainty = slope_uncertainty / (slope ** 2)

    except ValueError:
        coefficients = np.polyfit(x, y, deg=1, w=weights_polyfit)
        slope = coefficients[0]
        intercept = coefficients[1]

        # Direct calculation of slope and intercept for two data points
        slope = (y[-1] - y[0]) / (x[1] - x[0])
        intercept = y[0] - slope * x[0]
        # Error propagation for the slope and intercept
        slope_uncertainty = np.sqrt((y_err[0]**2 + y_err[1]**2) / (x[1] - x[0])**2)
        intercept_uncertainty = np.sqrt(
            y_err[0]**2 + (slope_uncertainty * x[0])**2 + (slope * x[0] * (y_err[1]**2 + y_err[0]**2) / (x[1] - x[0])**2)
        )
        # Error propagation for 1/slope
        inverse_slope_uncertainty = slope_uncertainty / (slope ** 2)

    fitted_line = slope * np.array(x) + intercept
    inv_slope = 1./slope

    # make plot
    plt.style.use(hep.style.ROOT)
    fig, ax = plt.subplots()
    plt.xlabel('Energy [$e^{-}$]')
    plt.ylabel('Threshold [DAC]')
    ax.errorbar(np.array(x), np.array(y), fmt='o', yerr=y_err)
    ax.plot(x, fitted_line, color='red', label='Fitted Line')
    ax.text(0.05, 0.95, f'Best fit: {slope:.3f} ± {slope_uncertainty:.3f} x + {intercept:.1f} ± {intercept_uncertainty:.1f}', ha='left', va='top', transform=plt.gca().transAxes, color='black')
    ax.text(0.05, 0.90, f'Calibration: {inv_slope:.1f} ± {inverse_slope_uncertainty:.1f} $e^{{-}}$ / DAC', ha='left', va='top', transform=plt.gca().transAxes, color='black')
    fig.tight_layout()
    fig.savefig(join(base_dir,f'calibration_plot.png'))
    plt.close()


if __name__ == "__main__":
    main()
