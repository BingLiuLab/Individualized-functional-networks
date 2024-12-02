"""
# @ creater by Qi Wang on 2023.2.16
# Description: Calculate delta LNC and network scale LNC values.
# Usage:
    A directory of database data results needs to be established and the location of the data to be matched should be 
    determined with the data/subject name as the unique index.
"""

import os
import numpy as np
import pandas as pd
import scipy.io as scio

def similarity(v_a, v_b):
    """Calculate the similarity between two vectors."""
    num = sum(v_a == v_b)
    return num / v_a.shape[0]

def load_network_labels(filepath):
    """Load network labels from a .mat file."""
    network_label = scio.loadmat(filepath)
    temp_label_lh = network_label['lh_labels'].flatten()
    temp_label_rh = network_label['rh_labels'].flatten()
    return np.hstack((temp_label_lh, temp_label_rh))

def process_subject(subject, type_label, save_path, s):
    """Process data for a single subject."""
    local_lh = pd.read_csv(f"{save_path}Orig2_{type_label}_homo2{s}lh.csv", header=None).values
    local_rh = pd.read_csv(f"{save_path}Orig2_{type_label}_homo2{s}rh.csv", header=None).values
    local = np.vstack((local_lh, local_rh))

    local_delta_lh = pd.read_csv(f"{save_path}local2_{type_label}_homo2{s}lh.csv", header=None).values
    local_delta_rh = pd.read_csv(f"{save_path}local2_{type_label}_homo2{s}rh.csv", header=None).values
    local_delta = np.vstack((local_delta_lh, local_delta_rh))

    label_lh = pd.read_csv(f"{save_path}final_iter_label{s}pial_lh.csv", header=None).values
    label_rh = pd.read_csv(f"{save_path}final_iter_label{s}pial_rh.csv", header=None).values
    label = np.vstack((label_lh, label_rh))

    return local, local_delta, label

def calculate_homogeneity(local, local_delta, label, temp_label):
    """Calculate homogeneity metrics for a subject."""
    n = local.shape[0]
    stats_homo = np.zeros(17)
    stats_homo_delta = np.zeros(17)
    stats_homo_abs_delta = np.zeros(17)
    num_homo = np.zeros(17)

    for i in range(n):
        if label[i] == 0:
            continue
        t = int(label[i] - 1)
        stats_homo[t] += local[i]
        stats_homo_delta[t] += local_delta[i]
        stats_homo_abs_delta[t] += abs(local_delta[i])
        num_homo[t] += 1

    res_homo = stats_homo / num_homo
    res_homo_delta = stats_homo_delta / num_homo
    res_homo_delta_abs = stats_homo_abs_delta / num_homo

    return res_homo, res_homo_delta, res_homo_delta_abs

def main(draw_path, work_path, type_label, network_label_path):
    """Main function to process all subjects and save results."""
    name = pd.read_csv(f"{work_path}HCP_test.csv", header=None)
    ls_base = name.values[:, 0]

    temp_label = load_network_labels(network_label_path)

    s = "_0.045_"
    res, res_del, res_del_abs = [], [], []
    homo_v, homo_v_delta = [], []

    for subject in ls_base:
        subject = str(int(subject))
        save_path = f"{work_path}Output/{subject}/"

        local, local_delta, label = process_subject(subject, type_label, save_path, s)
        homo_v.append(local.T.flatten())
        homo_v_delta.append(local_delta.T.flatten())

        res_homo, res_homo_delta, res_homo_delta_abs = calculate_homogeneity(local, local_delta, label, temp_label)

        res.append(res_homo)
        res_del.append(res_homo_delta)
        res_del_abs.append(res_homo_delta_abs)

    np.savetxt(f"{work_path}{type_label}_homo_vertex2{s}_20.csv", np.array(homo_v), delimiter=',', fmt='%.6f')
    np.savetxt(f"{work_path}{type_label}_homo_vertex_delta2{s}_20.csv", np.array(homo_v_delta), delimiter=',', fmt='%.6f')
    np.savetxt(f"{work_path}{type_label}_homo2{s}_20.csv", np.array(res), delimiter=',', fmt='%.6f')
    np.savetxt(f"{work_path}{type_label}_homo_delta2{s}_20.csv", np.array(res_del), delimiter=',', fmt='%.6f')

    print("Processing complete.")

if __name__ == "__main__":
    draw_path = "/mnt/data0/home/qwang/DATA/HCP/Output/"
    work_path = '/mnt/data0/home/qwang/DATA/HCP/'
    type_label = 'HCPtest_12_R1LR'
    network_label_path = '/mnt/data0/home/qwang/1000subjects_reference/1000subjects_clusters017_ref.mat'

    main(draw_path, work_path, type_label, network_label_path)