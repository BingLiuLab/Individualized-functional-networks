"""
# @ creater by Qi Wang on 2022.8.15
# Description: MRR calculation function. 
# Usage:
    By using a unique index of data/subject names, as the matching database data contains multiple runs 
    of data, even if the data to be matched is single run data, a run name needs to be set.
"""

import numpy as np
from scipy import stats
import pingouin as pg
import pandas as pd


def icc_calculate(Y, icc_type):
    """
    Calculate the Intraclass Correlation Coefficient (ICC).

    Parameters:
        Y (numpy.ndarray): Input data array.
        icc_type (str): Type of ICC to calculate.

    Returns:
        float: Calculated ICC value.
    """
    n, k = Y.shape
    
    mean_Y = np.mean(Y)
    SSC = ((np.mean(Y, axis=0) - mean_Y) ** 2).sum() * n
    SSR = ((np.mean(Y, axis=1) - mean_Y) ** 2).sum() * k
    SST = ((Y - mean_Y) ** 2).sum()
    SSE = SST - SSR - SSC

    MSE = SSE / ((n - 1) * (k - 1))
    MSC = SSC / (k - 1)
    MSR = SSR / (n - 1)

    if icc_type == "icc21":
        ICC = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)
    elif icc_type == "icc2k":
        ICC = (MSR - MSE) / (MSR + (MSC - MSE) / n)
    elif icc_type == "icc31":
        ICC = (MSR - MSE) / (MSR + (k - 1) * MSE)
    elif icc_type == "icc3k":
        ICC = (MSR - MSE) / MSR
    else:
        raise ValueError("Invalid ICC type.")

    return ICC

def reliability(x, alpha=4.3534):
    """
    Calculate reliability using a sigmoid function.

    Parameters:
        x (float): Input ICC value.
        alpha (float): Scaling factor.

    Returns:
        float: Reliability value.
    """
    return 1.0 / (1 + np.exp(-x * alpha))

def process_subject(subject, base_path,k=5.9890,hemi='rh'):
    """
    Process a single subject to calculate and save reliability values.

    Parameters:
        subject (str): Subject identifier.
        base_path (str): Base path for input and output files.

    Returns:
        None
    """
    subject = str(subject)


    file_paths = [
        f"{base_path}/R1_LR/{subject}/surf_{hemi}_time_pial.csv",
        f"{base_path}/R1_RL/{subject}/surf_{hemi}_time_pial.csv",
        f"{base_path}/R2_LR/{subject}/surf_{hemi}_time_pial.csv",
        f"{base_path}/R2_RL/{subject}/surf_{hemi}_time_pial.csv",
    ]

    data = [pd.read_csv(path, header=None).values for path in file_paths]
    n, m = data[0].shape

    reliabilities = [[] for _ in range(4)]

    for i in range(n):
        icc_results = []
        for j in range(len(data)):
            for l in range(j + 1, len(data)):
                combined_data = pd.concat([
                    pd.DataFrame({"value": data[j][i], "target": range(m), "class": (j + 1) * np.ones(m)}),
                    pd.DataFrame({"value": data[l][i], "target": range(m), "class": (l + 1) * np.ones(m)})
                ])
                icc = pg.intraclass_corr(
                    combined_data, targets="target", raters="class", ratings="value", nan_policy="omit"
                )["ICC"][1]
                icc_results.append(reliability(icc, k))

        reliabilities[0].append(np.mean(icc_results[:3]))
        reliabilities[1].append(np.mean(icc_results[1:4]))
        reliabilities[2].append(np.mean(icc_results[2:5]))
        reliabilities[3].append(np.mean(icc_results[3:]))

    for i, rel in enumerate(reliabilities):
        output_path = f"{base_path}/R{i + 1}_LR/{subject}/reliability_pial_0.01_{hemi}.csv"
        np.savetxt(output_path, rel, delimiter=",", fmt="%f")

    print(f"Processed subject {subject}, hemisphere {hemi}")

def main():
    base_path = '/mnt/data0/home/qwang/DATA/HCP/'
    subject_info = pd.read_csv(f"{base_path}/HCP_inf.csv")
    subjects = subject_info.values[:, 0]

    for subject in subjects:
        process_subject(subject, base_path,hemi='lh')

if __name__ == "__main__":
    main()


