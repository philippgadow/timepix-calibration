import argparse
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

from os import makedirs
from os.path import join
from hist import Hist
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths

from config import Config


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

    for measurement, hist in hists.items():
        hep.histplot(hist, histtype="step", label=labels[measurement], color=colours[measurement], ax=ax)

    if 'xlabel' in plotting_args: ax.set_xlabel(plotting_args['xlabel'])
    if 'ylabel' in plotting_args: ax.set_ylabel(plotting_args['ylabel'])
    if 'xlim' in plotting_args: ax.set_xlim(plotting_args['xlim'])
    if 'yscale' in plotting_args: ax.set_yscale(plotting_args['yscale'])
    
    plt.legend()
    fig.tight_layout()
    fig.savefig(output_name)
    plt.close()


def fitHistogram(hist, measurement, n_stddev, labels, plotting_args, output_name, save_plot=False):
    # values from histogram
    hist_data = hist.values()
    bin_centers = hist.axes[0].centers

    # get reasonable start values for fit
    peaks, _ = find_peaks(hist_data)
    highest_peak_index = np.argmax(hist_data[peaks])
    highest_peak_value = hist_data[highest_peak_index]
    highest_peak_bin_center = bin_centers[peaks][highest_peak_index]
    results_widths = peak_widths(hist_data, peaks, rel_height=0.5)
    fwhm = results_widths[0][highest_peak_index]

    initial_amplitude = highest_peak_value
    initial_mean = highest_peak_bin_center
    initial_sigma = fwhm / 2.35482004503

    # define fit function
    def gaussian(x, amplitude, mean, sigma):
        return amplitude * norm.pdf(x, loc=mean, scale=sigma)
    
    # truncate data for fit
    fit_range = (initial_mean - n_stddev * initial_sigma, initial_mean + n_stddev * initial_sigma)
    mask = (bin_centers < fit_range[0]) | (bin_centers > fit_range[1])
    filtered_hist = hist.copy()
    filtered_hist.values()[mask] = 0
    filtered_hist_data = filtered_hist.values()
    filtered_bin_centers = filtered_hist.axes[0].centers

    # fit function
    popt, pcov = curve_fit(gaussian, filtered_bin_centers, filtered_hist_data, p0=(initial_amplitude, initial_mean, initial_sigma), bounds=([0, fit_range[0], 0], [np.inf, fit_range[1], np.inf]))
    perr = np.sqrt(np.diag(pcov))
    threshold, threshold_err = popt[1], perr[1]

    # compute residuals and chi2
    fit_residuals = filtered_hist_data - gaussian(filtered_bin_centers, *popt)
    with np.errstate(divide='ignore', invalid='ignore'):
        chi_squared = np.sum(np.nan_to_num(((fit_residuals)**2 / filtered_hist), nan=0, posinf=0, neginf=0))

    if save_plot:
        plt.style.use(hep.style.ROOT)
        fig, ax = plt.subplots()
        plt.bar(bin_centers, hist_data, width=np.diff(bin_centers)[0], alpha=0.7, label=labels[measurement])
        plt.bar(filtered_bin_centers, filtered_hist_data, width=np.diff(filtered_bin_centers)[0], alpha=0.7, color='orange', label='Truncated histogram')
        plt.plot(bin_centers, gaussian(bin_centers, *popt), 'r-', label='Gaussian Fit')
        plt.text(0.95, 0.8, f'Best fit: {threshold:.2f} +/- {threshold_err:.2f}', transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', color='black')
        plt.text(0.95, 0.75, f'Chi-Squared: {chi_squared:.2f}', transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', color='black')
        if 'xlabel' in plotting_args: ax.set_xlabel(plotting_args['xlabel'])
        if 'ylabel' in plotting_args: ax.set_ylabel(plotting_args['ylabel'])
        if 'xlim' in plotting_args: ax.set_xlim(plotting_args['xlim'])
        if 'yscale' in plotting_args: ax.set_yscale(plotting_args['yscale'])
        plt.legend()
        fig.tight_layout()
        fig.savefig(output_name)
        plt.close()

    return threshold, threshold_err, chi_squared


def main(args=None):
    args = parse_args(args)
    config = Config(args.config_file)
    config.load()

    calibration =config.calibration

    labels =config.labels
    colours = config.colours
    plotting_args = config.plotting

    x_limit_low = int(plotting_args['xlim'][0])
    x_limit_high = int(plotting_args['xlim'][1])
    x_bins = x_limit_high - x_limit_low

    base_dir = join("output", "scans", config.name)
    plot_dir = join(base_dir, "plots")
    makedirs(plot_dir, exist_ok=True)

    # hists = {}

    # # collect histograms
    # for measurement in config.measurements:
    #     print(measurement)
    #     csvfile_name = join(base_dir, f'ths_counts_{measurement}.csv')

    #     data = np.genfromtxt(csvfile_name, delimiter=',')
    #     if len(data) == 0: continue
    #     thresholds = data[:,0]
    #     counts = data[:,1]

    #     hists[measurement] = Hist.new.Reg(x_bins, x_limit_low, x_limit_high, name="hist_count{measurement}").Double()
    #     for ths, cnt in zip(thresholds, counts):
    #         hists[measurement].fill(ths, weight=cnt)
    #     hists[measurement] /= np.max(hists[measurement].values())

    # # make plots
    # makePlot(hists, labels, colours, plotting_args, join(plot_dir, "ths_scan.png"))
    # for measurement in config.measurements:
    #     makedirs(join(plot_dir, measurement), exist_ok=True)
    #     makePlot({measurement: hists[measurement]}, labels, colours, plotting_args, join(plot_dir, measurement, f"ths_scan_{measurement}.png"))

    # # fit histograms
    # fitresults = {}

    # for measurement in config.measurements:
    #     makedirs(join(plot_dir, measurement), exist_ok=True)
    #     fit_results_threshold = []
    #     fit_results_threshold_err = []
    #     fit_results_invchi2 = []
    #     # do several fits restricted to certain ranges of the data to account for non-Gaussian tails
    #     for i_stddev in [.25, .5, .75, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3.0]:
    #         threshold, threshold_err, chi2 = fitHistogram(hists[measurement], measurement, i_stddev, labels, plotting_args, join(plot_dir, measurement, f"fit_ths_scan_{measurement}_{i_stddev}stddev.png"), save_plot=True)
    #         fit_results_threshold.append(threshold)
    #         fit_results_threshold_err.append(threshold_err)
    #         fit_results_invchi2.append(1. / chi2)

    #     # get weighted best fit value using inverse chi2 to provide means of weighting
    #     uncertainties = np.array(fit_results_threshold_err)
    #     weights = np.power(uncertainties, -2)
    #     fitresults[measurement] = {
    #         'threshold': np.average(fit_results_threshold, weights=fit_results_invchi2),
    #         'threshold_err_stat': np.sqrt(np.sum(weights * np.power(uncertainties, 2)) / np.sum(np.power(weights, 2))),
    #         'threshold_err_sys': np.std(fit_results_threshold),
    #     }

    # # save to csv file with numpy
    # data = []
    # for measurement, values in fitresults.items():
    #     energy_keV = float(calibration[measurement])
    #     energy_eh = energy_keV / 3.65 * 1000. # 3.65 eV to create e/h pair in Si
    #     row = [energy_keV, energy_eh, values['threshold'], values['threshold_err_stat'], values['threshold_err_sys']]
    #     data.append(row)

    # header = ['energy_keV', 'energy_eh', 'threshold', 'threshold_err_stat', 'threshold_err_sys']
    # data = np.array(data)
    # np.savetxt(join(base_dir, 'fitresults.csv'), data, delimiter=',', header=','.join(header), comments='', fmt='%s')

    data = np.genfromtxt(join(base_dir, 'fitresults.csv'), delimiter=',', skip_header=1)

    x = data[:,2]
    y = data[:,3]
    y_err_stat = data[:,4]
    y_err_syst = data[:,5]
    y_err_total = np.sqrt(np.power(np.array(y_err_stat), 2) + np.power(np.array(y_err_syst), 2))

    weights_polyfit = np.power(np.array(y_err_total), -1)
    coefficients = np.polyfit(x, y, deg=1, w=weights_polyfit)
    slope = coefficients[0]
    intercept = coefficients[1]
    fitted_line = slope * np.array(x) + intercept
    inv_slope = 1./slope


    # make plot
    plt.style.use(hep.style.ROOT)
    fig, ax = plt.subplots()
    plt.xlabel('Energy [$e^{-}$-hole pairs]')
    plt.ylabel('Threshold [DAC]')
    
    ax.errorbar(np.array(x), np.array(y), fmt='o', yerr=y_err_total)
    ax.plot(x, fitted_line, color='red', label='Fitted Line')
    ax.text(0.05, 0.95, f'Best fit: {slope:.3f} x + {intercept:.3f}', ha='left', va='top', transform=plt.gca().transAxes, color='black')
    ax.text(0.05, 0.90, f'Calibration: {inv_slope:.3f} $e^{{-}}$-hole pairs / DAC', ha='left', va='top', transform=plt.gca().transAxes, color='black')
    fig.tight_layout()
    fig.savefig(join(base_dir, 'calibration_plot.png'))
    plt.close()


if __name__ == "__main__":
    main()
