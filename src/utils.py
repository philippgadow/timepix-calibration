import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths

def makePlot(hists, labels, colours, plotting_args, output_name):
    plt.style.use(hep.style.ROOT)
    fig, ax = plt.subplots()

    for hist, label, colour in zip(hists, labels, colours):
        hep.histplot(hist, histtype="step", label=label, color=colour, ax=ax)

    if 'xlabel' in plotting_args: ax.set_xlabel(plotting_args['xlabel'])
    if 'ylabel' in plotting_args: ax.set_ylabel(plotting_args['ylabel'])
    if 'xlim' in plotting_args: ax.set_xlim(plotting_args['xlim'])
    if 'ylim' in plotting_args: ax.set_ylim(plotting_args['ylim'])
    if 'yscale' in plotting_args: ax.set_yscale(plotting_args['yscale'])
    
    plt.legend()
    fig.tight_layout()
    fig.savefig(output_name)
    plt.close()

def fitHistogramSigmoid(input_hist, properties, plotting_args, output_name):
    ### Fit a Sigmoid to the histogram and return the threshold value
    hist = input_hist.copy()
    hist.name = input_hist.name
    thresholds = np.array(hist.axes[0].centers)
    pixel_count_sums = np.array(hist.values())
    errors = np.sqrt(pixel_count_sums)

    # check if we should truncate histogram at a lowest threshold
    if 'lowest_threshold' in properties:
        lowest_threshold = int(properties['lowest_threshold'])
        mask = (thresholds < lowest_threshold)
        pixel_count_sums[mask] = 0
        errors[mask] = 0
        hist[...] = pixel_count_sums

    # Select the fitting range
    fit_low = int(properties['fitrange_sigmoid'][0])
    fit_high = int(properties['fitrange_sigmoid'][1])
    fit_mask = (thresholds >= fit_low) & (thresholds <= fit_high)
    fit_thresholds = thresholds[fit_mask]
    fit_pixel_counts = pixel_count_sums[fit_mask]
    fit_pixel_errors = errors[fit_mask]

    initial_guess_sigmoid = [np.max(fit_pixel_counts), np.mean(fit_thresholds), 1.0, np.min(fit_pixel_counts)]
    def sigmoid(x, L, x0, sigma, b):
        k = 1 / (sigma * np.sqrt(2 * np.log(2))) # compatible interpretation of sigma with the gaussian fit
        return L / (1 + np.exp(-k * (x - x0))) + b

    popt_sigmoid, pcov_sigmoid = curve_fit(sigmoid, fit_thresholds, fit_pixel_counts, p0=initial_guess_sigmoid)
    L_fit, x0_fit, sigma_fit, b_fit = popt_sigmoid
    perr_sigmoid = np.sqrt(np.diag(pcov_sigmoid))

    # Generate the s-curve for plotting
    x_fit_sigmoid = np.linspace(fit_low, fit_high, 300)
    y_fit_sigmoid = sigmoid(x_fit_sigmoid, *popt_sigmoid)

    chi_squared = 1. / np.sum((fit_pixel_counts - sigmoid(fit_thresholds, *popt_sigmoid))**2 / fit_pixel_counts)

    print("Sigmoid Fit Parameters:")
    print(f"L: {L_fit:.2f} ± {perr_sigmoid[0]:.2f}")
    print(f"Mean (μ): {x0_fit:.2f} ± {perr_sigmoid[1]:.2f}")
    print(f"Sigma (σ): {sigma_fit:.2f} ± {perr_sigmoid[2]:.2f}")
    print(f"Offset (b): {b_fit:.2f} ± {perr_sigmoid[3]:.2f}")

    # save plot
    plt.style.use(hep.style.ROOT)
    fig, ax = plt.subplots()
    plt.errorbar(thresholds, pixel_count_sums, yerr=errors, marker='o', linestyle='', capsize=5, color='black')
    plt.plot(x_fit_sigmoid, y_fit_sigmoid, color='orange', linestyle='--', label='S-curve fit')
    # Annotate the plot with the fit parameters
    textstr = '\n'.join((
        r'$\mu=%.1f \pm %.1f$' % (x0_fit, 0.1),
        r'$\sigma=%.1f \pm %.1f$' % (2*sigma_fit, perr_sigmoid[2])))
    plt.text(0.95, 0.75, textstr, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right',
            color='orange')
    plt.legend()
    if 'xlabel' in plotting_args: ax.set_xlabel(plotting_args['xlabel'])
    if 'ylabel' in plotting_args: ax.set_ylabel(plotting_args['ylabel'][0] + plotting_args['ylabel'][1:])
    if 'xlim' in plotting_args: ax.set_xlim(plotting_args['xlim'])
    if 'ylim' in plotting_args: ax.set_ylim(plotting_args['ylim'])
    if 'yscale' in plotting_args: ax.set_yscale(plotting_args['yscale'])
    plt.legend()
    fig.tight_layout()
    fig.savefig(output_name)
    plt.close()

    return x0_fit, perr_sigmoid[1], chi_squared

def fitHistogramGaussian(input_hist, n_stddev, bandwidth, properties, plotting_args, output_name):
    ### Fit a Gaussian to the derivative of histogram and return the threshold value
    # derivative of histogram which will be fitted
    hist = input_hist.derivative(bandwidth=int(bandwidth))
    hist.name = input_hist.name
    hist_data = hist.values()
    bin_centers = hist.axes[0].centers

    # check if we should truncate histogram at a lowest threshold
    if 'lowest_threshold' in properties:
        lowest_threshold = int(properties['lowest_threshold'])
        mask = (bin_centers < lowest_threshold)
        hist_data[mask] = 0
        hist[...] = hist_data

    # get reasonable start values for fit
    try:
        peaks, _ = find_peaks(hist_data, width=3)
        peaks_widths = peak_widths(hist_data, peaks, rel_height=0.7)
        rightmost_peak_index = peaks[-1]
    except IndexError:
        peaks, _ = find_peaks(hist_data, width=1)
        peaks_widths = peak_widths(hist_data, peaks, rel_height=0.5)

    # find the index of the rightmost peak
    try:
        rightmost_peak_index = peaks[-1]
        hist_bin_right = bin_centers[rightmost_peak_index]
        hist_data_right = hist_data[rightmost_peak_index]
        fwhm = peaks_widths[0][0]
    except IndexError:
        rightmost_peak_index = np.argmax(hist_data)
        hist_bin_right = bin_centers[rightmost_peak_index]
        hist_data_right = hist_data[rightmost_peak_index]
        fwhm = 1.
    

    initial_amplitude = hist_data_right
    initial_mean = hist_bin_right
    initial_sigma = fwhm / 2.35482004503

    print('Initial values for fit: ', initial_amplitude, initial_mean, initial_sigma)

    # define fit functions
    def gaussian(x, amplitude, mean, sigma):
        return amplitude * norm.pdf(x, loc=mean, scale=sigma)
    
    # truncate data for fit
    fit_range = (initial_mean - n_stddev * initial_sigma, initial_mean + n_stddev * initial_sigma)
    mask = (bin_centers < fit_range[0]) | (bin_centers > fit_range[1])
    filtered_hist = hist.copy()
    filtered_hist_data = filtered_hist.values()
    filtered_bin_centers = filtered_hist.axes[0].centers
    filtered_hist_data[mask] = 0
    filtered_hist[...] = filtered_hist_data

    # fit function
    popt, pcov = curve_fit(gaussian, filtered_bin_centers, filtered_hist_data, p0=(initial_amplitude, initial_mean, initial_sigma), bounds=([0, fit_range[0], 0], [np.inf, fit_range[1], np.inf]))

    perr = np.sqrt(np.diag(pcov))
    threshold, threshold_err = popt[1], perr[1]

    # compute residuals and chi2
    fit_residuals = filtered_hist_data - gaussian(filtered_bin_centers, *popt)
    with np.errstate(divide='ignore', invalid='ignore'):
        chi_squared = np.sum(np.nan_to_num(((fit_residuals)**2 / filtered_hist), nan=0, posinf=0, neginf=0))

    # save plot
    plt.style.use(hep.style.ROOT)
    fig, ax = plt.subplots()
    plt.bar(bin_centers, hist_data, width=np.diff(bin_centers)[0], alpha=0.7, label=properties['label'])
    plt.bar(filtered_bin_centers, filtered_hist_data, width=np.diff(filtered_bin_centers)[0], alpha=0.7, color='orange', label='Truncated histogram')
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
