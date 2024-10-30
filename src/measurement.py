from histogram import DifferentiableHist
from os import listdir, makedirs
from os.path import join, basename
import re
import pandas as pd
import shutil
from itertools import product
import rich.progress
import logging
import uproot
import numpy as np
from utils import makePlot, fitHistogramSigmoid, fitHistogramGaussian

class Measurement:
    def __init__(self, name, data):
        self.name = name
        self.group = data.get('group', None)
        self.type = data.get('type', None)
        self.properties = data.get('properties', {})

    def __str__(self):
        return f'Measurement(name={self.name}, group={self.group}, type={self.type}, properties={self.properties})'

    def extract_measurement(self, base_dir):
        """Needs to be implemented by subclasses"""
        logging.error('extract_measurement not implemented for ' + self.name)
        pass

    def collect_data(self, base_dir, plotting_args):
        """Needs to be implemented by subclasses"""
        logging.error('collect_data not implemented for ' + self.name)
        pass


class BaselineMeasurement(Measurement):
    def __init__(self, name, data):
        super().__init__(name, data)

    def extract_measurement(self, base_dir):
        pass

    def collect_data(self, base_dir, plotting_args):
        calibration_data = {}
        calibration_data['energy'] = self.properties.get('calibration_energy_keV', None)
        calibration_data['threshold'] = self.properties.get('peak_mean', None)
        calibration_data['threshold_width'] = self.properties.get('peak_width', None)
        return calibration_data


class SpectrumMeasurement(Measurement):
    def __init__(self, name, data):
        super().__init__(name, data)
        self.data_tag = f'{self.group}_{self.name}'

        self.input_dir = data.get('properties', {}).get('input_dir', None)
        self.measurement_dir = None
        self.outputfile_counts_name = None

    def copy_input_files(self, base_dir):
        self.measurement_dir = join(base_dir, self.data_tag, "input_files")
        try:
            listdir(self.measurement_dir)
        except FileNotFoundError:
            shutil.copytree(self.input_dir, self.measurement_dir)

    def process_file_csv(self, filename):
        # get threshold from filename
        threshold = re.search(r'\d+', basename(filename)).group()

        # get number of single pixel hits from corry clustering results
        # open csv file with pandas, ignore first 7 rows, ignore comments starting with "====", split by ","
        # header is (row, column, acq_mode, value)
        data = pd.read_csv(filename, skiprows=7, comment='=', sep=',', names=['row', 'column', 'acq_mode', 'count'])
        n_single_pixel_clusters = data['count'].sum()
        
        # restrict n_bins to 256, no long_count mode for H2M
        n_bins = 256
        bin_edges = np.zeros(n_bins + 1, dtype=float)
        bin_values = np.zeros(n_bins, dtype=float)

        return threshold, n_single_pixel_clusters, bin_edges, bin_values
    
    def process_file_root(self, filename):
        # get threshold from filename
        threshold = re.search(r'\d+', basename(filename)).group()

        # get number of hits from histogram
        root_file = uproot.open(filename)
        histogram = root_file["Pixel Count Sum"]
        if not histogram:
            logging.error(f"Warning: Histogram 'Pixel Count Sum' not found in {filename}. Skipping...")
            return
        n_single_pixel_clusters = histogram.values().sum()
        bin_edges = histogram.edges()
        bin_values = histogram.values()

        return threshold, n_single_pixel_clusters, bin_edges, bin_values

    def extract_measurement(self, base_dir):
        # copy input files to output directory and use the copies for processing
        self.copy_input_files(base_dir)

        print(f'Processing files in {self.measurement_dir}')
        self.outputfile_counts_name = join(base_dir, self.data_tag, f'ths_counts_{self.data_tag}.csv')

        with open(self.outputfile_counts_name, 'w') as file_count:
            with rich.progress.Progress() as progress:
                task = progress.add_task("[cyan]Processing files...", total=len(listdir(join(base_dir, self.data_tag))))
                for f in sorted(listdir(self.measurement_dir)):
                    progress.update(task, advance=1)
                    try:
                        if f.endswith('.csv'):
                            threshold, n_single_pixel_cluster, bin_edges, bin_values = self.process_file_csv(join(self.measurement_dir, f))
                        elif f.endswith('.root'):
                            threshold, n_single_pixel_cluster, bin_edges, bin_values = self.process_file_root(join(self.measurement_dir, f))
                        else:
                            logging.error('Unknown file type ' + f)
                            continue
                    except OSError:
                        logging.error('Could not open file ' + f)
                        continue
                    except AttributeError:
                        logging.error('Problem opening histogram in file ' + f)
                        continue
                    
                    line = f"{threshold},{n_single_pixel_cluster}\n"
                    file_count.write(line)
                
        # sort outputfile_counts_name by threshold
        header = f'# {self.name}/{self.name}\n#threshold,counts'
        df = pd.read_csv(self.outputfile_counts_name, names=['threshold', 'counts'])
        df = df.sort_values(by=['threshold'])
        df.to_csv(self.outputfile_counts_name, index=False, header=header)

    def collect_data(self, base_dir, plotting_args):
        # global binning
        x_limit_low = int(plotting_args['xlim'][0])
        x_limit_high = int(plotting_args['xlim'][1])
        x_bins = x_limit_high - x_limit_low + 1

        self.outputfile_counts_name = join(base_dir, self.data_tag, f'ths_counts_{self.data_tag}.csv')
        print(self.outputfile_counts_name)
        try:
            data = np.genfromtxt(self.outputfile_counts_name, delimiter=',')
        except FileNotFoundError:
            print(f'Could not find {self.outputfile_counts_name}')
        
        if len(data) == 0: return
        thresholds = data[:,0]
        counts = data[:,1]

        # get histogram
        hist = DifferentiableHist.new.Reg(x_bins, x_limit_low, x_limit_high, name=f"hist_count{self.name}").Double()
        for ths, cnt in zip(thresholds, counts):
            hist.fill(ths, weight=cnt)
        # get normalised histogram with maximum value of 1
        norm_hist = hist.copy()
        norm_hist /= np.max(hist.values())

        # make plot of raw histogram
        plot_dir = join(base_dir, self.data_tag, "plots_cnt")
        makedirs(plot_dir, exist_ok=True)
        makePlot([hist], [self.properties['label']], [self.properties['colour']], plotting_args, join(plot_dir, f"ths_scan_{self.data_tag}.png"))

        # plot first and second derivatives
        for bandwidth in self.properties['derivative_bandwith']:
            deriv1 = hist.derivative(bandwidth=int(bandwidth))
            makePlot([deriv1], [self.properties['label']], [self.properties['colour']], plotting_args, join(plot_dir, f"deriv1_ths_scan_{self.data_tag}_bandwidth{bandwidth}.png"))
            deriv2 = deriv1.derivative(bandwidth=int(bandwidth))
            makePlot([deriv2], [self.properties['label']], [self.properties['colour']], plotting_args, join(plot_dir, f"deriv2_ths_scan_{self.data_tag}_bandwidth{bandwidth}.png"))

        # perform fits
        fit_results_threshold = []
        fit_results_threshold_err = []
        fit_results_invchi2 = []

        # do several fits restricted to certain ranges of the data to account for non-Gaussian tails
        range_stddev = self.properties['fitrange_nstddev']
        range_bandwidth = self.properties['derivative_bandwith']
        if not type(range_bandwidth) == list: range_bandwidth = [range_bandwidth]

        # perform fit of Sigmoid for histogram
        threshold, threshold_err, chi2 = fitHistogramSigmoid(hist, self.properties, plotting_args, join(plot_dir, f"fit_ths_scan_sigmoid_{self.data_tag}.png"))

        # perform fits of Gaussian for derivative of normalised histogram
        for i_stddev, bandwith in product(range_stddev, range_bandwidth):
            threshold, threshold_err, chi2 = fitHistogramGaussian(norm_hist, i_stddev, bandwith, self.properties, plotting_args, join(plot_dir, f"fit_ths_scan_{self.data_tag}_{i_stddev}stddev_{bandwith}derivbandwidth.png"))
            fit_results_threshold.append(threshold)
            fit_results_threshold_err.append(threshold_err)
            fit_results_invchi2.append(1. / chi2)

        # get weighted best fit value using inverse chi2 to provide means of weighting
        calibration_data = {}
        calibration_data['energy'] = self.properties.get('calibration_energy_keV', None)
        calibration_data['threshold'] = np.average(fit_results_threshold, weights=fit_results_invchi2)
        calibration_data['threshold_width'] = np.std(fit_results_threshold)
        return calibration_data