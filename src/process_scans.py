import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep

from itertools import product
from os import makedirs
from os.path import join
from hist import Hist
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths

from config import Config


class DifferentiableHist(Hist):
        def derivative(self, bandwidth=10, mirror=True):
            # Ensure the histogram is 1D
            if len(self.axes) != 1:
                raise ValueError("This method only works for 1D histograms.")

            # Get histogram values
            values = self.view()

            # Compute the moving average
            smoothed = np.convolve(values, np.ones(bandwidth)/bandwidth, 'same')

            # Compute the smoothed derivative
            deriv = np.diff(smoothed)

            # Append the last bin value to maintain bin count
            deriv = np.append(deriv, deriv[-1])

            # Mirror histogram at y-axis (required if derivative is negative and we want to fit positive peaks later)
            if mirror: deriv *= -1
            
            # Construct a new histogram for the derivative
            new_hist = self.copy()
            new_hist[:] = deriv

            return new_hist
            

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


def makePlot(hists, labels, colours, plotting_args, output_name):
    plt.style.use(hep.style.ROOT)
    fig, ax = plt.subplots()

    for data_tag, hist in hists.items():
        # shitty way of coding, I know...
        measurement = data_tag.replace('_all', '').replace('_odd', '').replace('_even', '')
        hep.histplot(hist, histtype="step", label=labels[measurement], color=colours[measurement], ax=ax)

    if 'xlabel' in plotting_args: ax.set_xlabel(plotting_args['xlabel'])
    if 'ylabel' in plotting_args: ax.set_ylabel(plotting_args['ylabel'])
    if 'xlim' in plotting_args: ax.set_xlim(plotting_args['xlim'])
    if 'ylim' in plotting_args: ax.set_ylim(plotting_args['ylim'])
    if 'yscale' in plotting_args: ax.set_yscale(plotting_args['yscale'])
    
    plt.legend()
    fig.tight_layout()
    fig.savefig(output_name)
    plt.close()


def fitHistogram(input_hist, measurement, n_stddev, bandwidth, fitting_args, labels, plotting_args, output_name, fit_double_gaussian=True):
    # derivative of histogram which will be fitted
    hist = input_hist.derivative(bandwidth=int(bandwidth))
    hist.name = input_hist.name
    hist_data = hist.values()
    bin_centers = hist.axes[0].centers

    # check if we should truncate histogram at a lowest threshold
    if measurement in fitting_args['lowest_threshold']:
        lowest_threshold = int(fitting_args['lowest_threshold'][measurement])
        mask = (bin_centers < lowest_threshold)
        hist_data[mask] = 0
        hist[...] = hist_data

    # get reasonable start values for fit
    try:
        peaks, _ = find_peaks(hist_data, width=4)
        peaks_widths = peak_widths(hist_data, peaks, rel_height=0.7)
        rightmost_peak_index = peaks[-1]
    except IndexError:
        peaks, _ = find_peaks(hist_data, width=1)
        peaks_widths = peak_widths(hist_data, peaks, rel_height=0.5)
        rightmost_peak_index = peaks[-1]

    # find the index of the rightmost peak
    rightmost_peak_index = peaks[-1]
    hist_bin_right = bin_centers[rightmost_peak_index]
    hist_data_right = hist_data[rightmost_peak_index]
    fwhm = peaks_widths[0][0]

    initial_amplitude = hist_data_right
    initial_mean = hist_bin_right
    initial_sigma = fwhm / 2.35482004503

    print('Initial values for fit: ', initial_amplitude, initial_mean, initial_sigma)

    # define fit functions
    def gaussian(x, amplitude, mean, sigma):
        return amplitude * norm.pdf(x, loc=mean, scale=sigma)
    
    def double_gaussian(x, a1, mu1, sigma1, a2, mu2, sigma2):
        gauss1 = a1 * norm.pdf(x, mu1, sigma1)
        gauss2 = a2 * norm.pdf(x, mu2, sigma2)
        return gauss1 + gauss2

    # truncate data for fit
    fit_range = (initial_mean - n_stddev * initial_sigma, initial_mean + n_stddev * initial_sigma)
    mask = (bin_centers < fit_range[0]) | (bin_centers > fit_range[1])
    # do not restrict fit to the right if fitting double gaussian
    if fit_double_gaussian: mask = (bin_centers < fit_range[0])
    filtered_hist = hist.copy()
    filtered_hist_data = filtered_hist.values()
    filtered_bin_centers = filtered_hist.axes[0].centers
    filtered_hist_data[mask] = 0
    filtered_hist[...] = filtered_hist_data

    # fit function
    if fit_double_gaussian:
        # fit both k alpha and k beta line: k beta is always to the right of k alpha and has less than 20% intensity
        # https://xdb.lbl.gov/Section1/Table_1-3.pdf
        popt, pcov = curve_fit(
            double_gaussian, filtered_bin_centers, filtered_hist_data,
            p0=(initial_amplitude, initial_mean, initial_sigma, initial_amplitude*0.2, initial_mean+1, initial_sigma),
            bounds=([0, fit_range[0], 0, 0, initial_mean, 0], [np.inf, np.inf, np.inf, initial_amplitude*.4, np.inf, np.inf]),
        )
    else:
        popt, pcov = curve_fit(gaussian, filtered_bin_centers, filtered_hist_data, p0=(initial_amplitude, initial_mean, initial_sigma), bounds=([0, fit_range[0], 0], [np.inf, fit_range[1], np.inf]))

    perr = np.sqrt(np.diag(pcov))
    threshold, threshold_err = popt[1], perr[1]

    # compute residuals and chi2
    if fit_double_gaussian:
        fit_residuals = filtered_hist_data - double_gaussian(filtered_bin_centers, *popt)
    else:
        fit_residuals = filtered_hist_data - gaussian(filtered_bin_centers, *popt)
    with np.errstate(divide='ignore', invalid='ignore'):
        chi_squared = np.sum(np.nan_to_num(((fit_residuals)**2 / filtered_hist), nan=0, posinf=0, neginf=0))

    # save plot
    plt.style.use(hep.style.ROOT)
    fig, ax = plt.subplots()
    plt.bar(bin_centers, hist_data, width=np.diff(bin_centers)[0], alpha=0.7, label=labels[measurement])
    plt.bar(filtered_bin_centers, filtered_hist_data, width=np.diff(filtered_bin_centers)[0], alpha=0.7, color='orange', label='Truncated histogram')
    if fit_double_gaussian:
        plt.plot(bin_centers, double_gaussian(bin_centers, *popt), 'r-', label='Double Gaussian Fit')
        plt.plot(bin_centers, gaussian(bin_centers, *popt[0:3]), 'g--', label='Gaussian Fit 1')
        plt.plot(bin_centers, gaussian(bin_centers, *popt[3:]), 'b--', label='Gaussian Fit 2')
        y_offset = 0.1
    else: 
        plt.plot(bin_centers, gaussian(bin_centers, *popt), 'r-', label='Gaussian Fit')
        y_offset = 0.
    plt.text(0.95, 0.8 - y_offset, f'Best fit: {threshold:.2f} +/- {threshold_err:.2f}', transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', color='black')
    plt.text(0.95, 0.75 - y_offset, f'Chi-Squared: {chi_squared:.3f}', transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', color='black')
    plt.text(0.05, 0.95, f'Moving average\nderivative: {bandwidth} bins', transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    if 'xlabel' in plotting_args: ax.set_xlabel(plotting_args['xlabel'])
    if 'ylabel' in plotting_args: ax.set_ylabel('Derivative of ' + plotting_args['ylabel'][0].lower() + plotting_args['ylabel'][1:])
    if 'xlim' in plotting_args: ax.set_xlim(plotting_args['xlim'])
    if 'ylim' in plotting_args: ax.set_ylim(plotting_args['ylim'])
    if 'yscale' in plotting_args: ax.set_yscale(plotting_args['yscale'])
    ax.set_ylim([0, 1.3 * np.max(hist_data)])
    plt.legend()
    fig.tight_layout()
    fig.savefig(output_name)
    plt.close()

    return threshold, threshold_err, chi_squared


def main(args=None):
    args = parse_args(args)
    config = Config(args.config_file)
    config.load()
    
    # process measurement groups
    for measurement_group_name, measurements in config.measurements.items():
        process_measurement_group(measurement_group_name, measurements, config)


def process_measurement_group(measurement_group_name, measurements, config):
    calibration =config.calibration
    labels =config.labels
    colours = config.colours
    plotting_args = config.plotting
    fitting_args = config.fitting

    x_limit_low = int(plotting_args['xlim'][0])
    x_limit_high = int(plotting_args['xlim'][1])
    x_bins = x_limit_high - x_limit_low + 1

    base_dir = join("output", "scans", config.name, config.timestamp, measurement_group_name)
    plot_dir = join(base_dir, "plots_cnt")
    makedirs(plot_dir, exist_ok=True)

    # collect histograms
    hists = {}
    for measurement, column in product(measurements, config.columns):
        data_tag = f'{measurement}_{column}'
        print(f'Collecting data for {data_tag}')
        csvfile_name = join(base_dir, f'ths_counts_{measurement_group_name}_{data_tag}.csv')

        data = np.genfromtxt(csvfile_name, delimiter=',')
        if len(data) == 0: continue
        thresholds = data[:,0]
        counts = data[:,1]

        hists[data_tag] = DifferentiableHist.new.Reg(x_bins, x_limit_low, x_limit_high, name="hist_count{data_tag}").Double()
        for ths, cnt in zip(thresholds, counts):
            hists[data_tag].fill(ths, weight=cnt)
        hists[data_tag] /= np.max(hists[data_tag].values())

        # make plots
        makedirs(join(plot_dir, data_tag), exist_ok=True)
        makePlot({data_tag: hists[data_tag]}, labels, colours, plotting_args, join(plot_dir, data_tag, f"ths_scan_{data_tag}.png"))

    # make plots of all measurements
    for column in config.columns:
        hists_column = {h: d for h, d in hists.items() if column in h}
        makePlot(hists_column, labels, colours, plotting_args, join(plot_dir, f"ths_scan_{column}.png"))


    # fit histograms
    fitresults = {}
    for measurement, column in product(measurements, config.columns):
        print('Fitting ',measurement, column)
        data_tag = f'{measurement}_{column}'
        makedirs(join(plot_dir, data_tag), exist_ok=True)
        fit_results_threshold = []
        fit_results_threshold_err = []
        fit_results_invchi2 = []

        # do several fits restricted to certain ranges of the data to account for non-Gaussian tails
        fit_double_gaussian = fitting_args['fit_double_gaussian'][measurement]
        range_stddev = fitting_args['fitrange_nstddev'][measurement]
        range_bandwidth = fitting_args['derivative_bandwith'][measurement]
        if not type(range_bandwidth) == list: range_bandwidth = [range_bandwidth]

        # check if we should fit double gaussian
        for i_stddev, bandwith in product(range_stddev, range_bandwidth):
            threshold, threshold_err, chi2 = fitHistogram(hists[data_tag], measurement, i_stddev, bandwith, fitting_args, labels, plotting_args, join(plot_dir, data_tag, f"fit_ths_scan_{data_tag}_{i_stddev}stddev_{bandwith}derivbandwidth.png"), fit_double_gaussian=fit_double_gaussian)
            fit_results_threshold.append(threshold)
            fit_results_threshold_err.append(threshold_err)
            fit_results_invchi2.append(1. / chi2)

        # get weighted best fit value using inverse chi2 to provide means of weighting
        uncertainties = np.array(fit_results_threshold_err)
        weights = np.power(uncertainties, -2)
        fitresults[data_tag] = {
            'threshold': np.average(fit_results_threshold, weights=fit_results_invchi2),
            'threshold_err_stat': np.sqrt(np.sum(weights * np.power(uncertainties, 2)) / np.sum(np.power(weights, 2))),
            'threshold_err_sys': np.std(fit_results_threshold),
        }

    # final calibration result
    for column in config.columns:
        data = []
        for data_tag, values in fitresults.items():
            if column not in data_tag: continue
            # shitty way of coding, I know...
            measurement = data_tag.replace('_all', '').replace('_odd', '').replace('_even', '')
            if measurement in config.overwrite_threshold:
                print(f'Overwriting threshold for {measurement} with {config.overwrite_threshold[measurement]}')
                values['threshold'] = config.overwrite_threshold[measurement][0]
                values['threshold_err_stat'] = config.overwrite_threshold[measurement][1]
                values['threshold_err_sys'] = 0.
            energy_keV = float(calibration[measurement])
            energy_eh = energy_keV / 3.65 * 1000. # 3.65 eV to create e/h pair in Si
            row = [energy_keV, energy_eh, values['threshold'], values['threshold_err_stat'], values['threshold_err_sys']]
            data.append(row)

        header = ['energy_keV', 'energy_eh', 'threshold', 'threshold_err_stat', 'threshold_err_sys']
        data = np.array(data)
        np.savetxt(join(base_dir, f'fitresults_{column}.csv'), data, delimiter=',', header=','.join(header), comments='', fmt='%s')
        # data = np.genfromtxt(join(base_dir, f'fitresults_{column}.csv'), delimiter=',', skip_header=1)

        x = data[:,1]
        y = data[:,2]
        y_err_stat = data[:,3]
        y_err_syst = data[:,4]
        y_err_total = np.sqrt(np.power(y_err_stat, 2) + np.power(y_err_syst, 2))

        weights_polyfit = np.power(np.array(y_err_total), -1)

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
            slope_uncertainty = np.sqrt((y_err_total[0]**2 + y_err_total[1]**2) / (x[1] - x[0])**2)
            intercept_uncertainty = np.sqrt(
                y_err_total[0]**2 + (slope_uncertainty * x[0])**2 + (slope * x[0] * (y_err_total[1]**2 + y_err_total[0]**2) / (x[1] - x[0])**2)
            )
            # Error propagation for 1/slope
            inverse_slope_uncertainty = slope_uncertainty / (slope ** 2)

        fitted_line = slope * np.array(x) + intercept
        inv_slope = 1./slope

        # make plot
        plt.style.use(hep.style.ROOT)
        fig, ax = plt.subplots()
        plt.xlabel('Energy [$e^{-}]')
        plt.ylabel('Threshold [DAC]')
        ax.errorbar(np.array(x), np.array(y), fmt='o', yerr=y_err_total)
        ax.plot(x, fitted_line, color='red', label='Fitted Line')
        ax.text(0.05, 0.95, f'Best fit: {slope:.3f} ± {slope_uncertainty:.3f} x + {intercept:.1f} ± {intercept_uncertainty:.1f}', ha='left', va='top', transform=plt.gca().transAxes, color='black')
        ax.text(0.05, 0.90, f'Calibration: {inv_slope:.1f} ± {inverse_slope_uncertainty:.1f} $e^{{-}}$ / DAC', ha='left', va='top', transform=plt.gca().transAxes, color='black')
        fig.tight_layout()
        fig.savefig(join(base_dir, f'calibration_plot_{column}.png'))
        plt.close()


if __name__ == "__main__":
    main()
