"""
This script computes 95% confidence intervals for Pearson correlation
coefficients using Fisher's z transformation. It takes a list of correlation
records containing Pearson r, p-value, and sample size, converts each r value to
Fisher z space, computes the confidence interval in z space, transforms the
bounds back to the correlation scale, and prints formatted statistical summaries.
"""

import numpy as np


CORRELATION_RECORDS = [
    {"r": -0.308871135910392, "p": 0.0133357984450564, "n": 90},
    {"r": -0.401419826887844, "p": 1.12319603850036e-06, "n": 180},
    {"r": 0.386392470746084, "p": 2.34e-07, "n": 180}
]


def calculate_correlation_ci(r_value, n_samples, confidence_z=1.96):
    if n_samples <= 3:
        return np.nan, np.nan

    r_value = np.clip(r_value, -0.999999, 0.999999)

    fisher_z = 0.5 * np.log((1 + r_value) / (1 - r_value))
    standard_error = 1 / np.sqrt(n_samples - 3)

    z_lower = fisher_z - confidence_z * standard_error
    z_upper = fisher_z + confidence_z * standard_error

    r_lower = np.tanh(z_lower)
    r_upper = np.tanh(z_upper)

    return r_lower, r_upper


for index, record in enumerate(CORRELATION_RECORDS, start=1):
    r_value = record["r"]
    p_value = record["p"]
    n_samples = record["n"]

    r_lower, r_upper = calculate_correlation_ci(
        r_value,
        n_samples
    )

    print(f"Dataset {index}:")
    print(
        f"Pearson r = {r_value:.3f}, "
        f"95% CI = [{r_lower:.3f}, {r_upper:.3f}], "
        f"p = {p_value:.3e}, "
        f"n = {n_samples}"
    )
    print()