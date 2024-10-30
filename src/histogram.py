import numpy as np
from hist import Hist

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
        