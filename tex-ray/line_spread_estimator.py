import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import convolve1d

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

"""
This file contains a small script for estimating the line spread function from
an straight edge experiment (edge spread). The function also allows for
estimating an equivalent kernel for the system with binning.
"""


def gaussian(x, mu, sigma):
    y = np.exp(-((x - mu) ** 2) / (sigma**2)) / (sigma * np.sqrt(np.pi))
    return y


def convolve_gaussian(x0, x, sigma):
    y = np.exp(-((x / sigma) ** 2))
    y /= np.sum(y)
    return convolve1d(x0, y)


def bin_line(x, binning_number):
    x_binned = (
        x.reshape(len(x) // binning_number, binning_number).sum(1)
        / binning_number
    )
    return x_binned


if __name__ == "__main__":
    edge_test_path = "./tex-ray/edge_spread.csv" # format: pixel, %intensity
    sign_conv = -1  # Depends on what side is bright
    guess_experiment = [53.5, 4.0]
    L = 100  # Reasonable domain size
    l = 20  # Reasonable kernel size
    binning_number = 4
    guess_simulation = [0.7]

    edge_spread = np.loadtxt(edge_test_path, delimiter=",")
    line_spread = sign_conv * (edge_spread[1:, 1] - edge_spread[0:-1, 1])
    line_spread /= np.sum(line_spread)
    x = np.arange(0, len(line_spread))

    popt_exp, pcov_exp = curve_fit(
        gaussian, x, line_spread, p0=guess_experiment
    )

    plt.figure(figsize=(4.77, 3.58))
    plt.plot(x, line_spread, "b+:", label="Data")
    plt.plot(x, gaussian(x, *popt_exp), "ro-", label="Fit")
    plt.axis([48, 58, 0, 0.3])
    plt.xticks(fontsize=11)
    plt.xlabel("Pixel along sampled line s(-)")
    plt.yticks(fontsize=11)
    plt.ylabel("Normalized intensity (arb)")
    plt.title("Estimate of line spread function", fontsize=11)
    plt.legend()
    plt.savefig("./tex-ray/line_spread.pdf")
    plt.show()

    x_kernel = np.arange(-l, l + 1)
    simulated_edge = np.zeros(2 * L)
    simulated_edge[L:] = 1.0
    simulated_line_spread = gaussian(x_kernel, 0, popt_exp[1])
    simulated_edge_spread = convolve1d(simulated_edge, simulated_line_spread)
    simulated_edge_spread_binned = bin_line(
        simulated_edge_spread, binning_number
    )
    binned_simulated_edge = bin_line(simulated_edge, binning_number)

    popt_sim, pcov_sim = curve_fit(
        lambda x, sigma: convolve_gaussian(binned_simulated_edge, x, sigma),
        x_kernel,
        simulated_edge_spread_binned,
        p0=guess_simulation,
    )

    print(
        "Estimated standard deviation of LSF for binned system: ", popt_sim[0]
    )
