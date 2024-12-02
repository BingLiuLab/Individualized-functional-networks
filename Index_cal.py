"""
# @ creater by Qi Wang on 2023.1.29
# Description: Local Network Complexity.
# Usage:
    A directory of database data results needs to be established and the location of the data to be matched should be 
    determined with the data/subject name as the unique index.Finally generate LNC values for each individual vertex by vertex.
"""


import scipy.io as scio
import numpy as np
import random
import csv
import os
import pandas as pd
import math

def similarity(v_a,v_b):
    num=0
    for i in range(v_a.shape[0]):
        if v_a[i]==v_b[i]:
            num+=1
    return num/v_a.shape[0]

# Yeo 17 network calculate
    # Orig_homo=np.array(list(Temp_Label))
    # Orig_homo=np.zeros(n)
    # for i in range(n):
        
    #     if Temp_Label[i]==0:
    #         # Local_homo[i]=0
    #         Orig_homo[i]=0
    #         continue
    #     adjacent=Temp_Label[ad[i]-1]
    #         # Get unique elements and their counts
    #     unq,counts = np.unique(adjacent,return_counts=True)
    #     temp=0
    #     N=m
        
    #     for j in range(unq.shape[0]):
    #         if unq[j]==0:
    #             # counts[np.argmax(counts)]+=counts[j]
    #             N=m-counts[j]
    #             continue    
    #         temp+=math.pow(counts[j],2)
    #     t=math.sqrt(temp)/N
    
    #         # Local_homo[i]=(t-math.sqrt(23)/19)/(1-math.sqrt(23)/19)-Temp_local[i]
    #         # Orig_homo[i]=(t-math.sqrt(23)/19)/(1-math.sqrt(23)/19)

    #         # Local_homo[i]=Temp_local[i]-(t-math.sqrt(23)/19)/(1-math.sqrt(23)/19)
    #     Orig_homo[i]=1-(t-math.sqrt(23)/19)/(1-math.sqrt(23)/19)
            
    # np.savetxt(work_path+'Temp_homo2_'+hemi+'.csv',Orig_homo,delimiter=',',fmt='%.6f')

import scipy.io as scio
import numpy as np
import pandas as pd
import os
import math

def similarity(v_a, v_b):
    """
    Calculate the similarity between two vectors.

    Args:
        v_a (numpy.ndarray): First vector.
        v_b (numpy.ndarray): Second vector.

    Returns:
        float: Similarity value between 0 and 1.
    """
    num = np.sum(v_a == v_b)
    return num / v_a.shape[0]

def load_network_labels(filepath, hemi):
    """
    Load network labels for a specific hemisphere.

    Args:
        filepath (str): Path to the .mat file containing network labels.
        hemi (str): Hemisphere ('lh' or 'rh').

    Returns:
        numpy.ndarray: Flattened array of network labels.
    """
    Network_Label = scio.loadmat(filepath)
    Temp_Label = Network_Label[f'{hemi}_labels']
    return Temp_Label.flatten()

def calculate_local_homogeneity(ad, label, temp_label, temp_local, n, m):
    """
    Calculate local and original homogeneity for a subject.

    Args:
        ad (numpy.ndarray): Adjacency matrix.
        label (numpy.ndarray): Subject label data.
        temp_label (numpy.ndarray): Template labels.
        temp_local (numpy.ndarray): Template local data.
        n (int): Number of nodes.
        m (int): Number of neighbors.

    Returns:
        tuple: Arrays for local and original homogeneity.
    """
    Local_homo = np.zeros(n)
    Orig_homo = np.zeros(n)

    for i in range(n):
        if temp_label[i] == 0:
            Local_homo[i] = 0
            Orig_homo[i] = 0
            continue

        adjacent = label[ad[i] - 1]
        unq, counts = np.unique(adjacent, return_counts=True)
        temp = 0
        N = m

        for j in range(unq.shape[0]):
            if unq[j] == 0:
                N = m - counts[j]
                continue
            temp += math.pow(counts[j], 2)

        t = math.sqrt(temp) / N
        Orig_homo[i] = 1 - (t - math.sqrt(23) / 19) / (1 - math.sqrt(23) / 19)
        Local_homo[i] = Orig_homo[i] - temp_local[i]

    return Local_homo, Orig_homo


def process_subject_data(work_path, subject, hemi, temp_label, temp_local, ad, n, m, run):
    """
    Process data for a single subject and save the results.

    Args:
        work_path (str): Working directory path.
        subject (str): Subject ID.
        hemi (str): Hemisphere ('lh' or 'rh').
        temp_label (numpy.ndarray): Template labels.
        temp_local (numpy.ndarray): Template local data.
        ad (numpy.ndarray): Adjacency matrix.
        n (int): Number of nodes.
        m (int): Number of neighbors.
        run (str): Run identifier.
    """
    save_path = os.path.join(work_path, "Output", subject)
    label = pd.read_csv(os.path.join(save_path, f"final_iter_label_0.045_pial_{hemi}.csv"), header=None).values

    Local_homo, Orig_homo = calculate_local_homogeneity(ad, label, temp_label, temp_local, n, m)

    np.savetxt(
        os.path.join(save_path, f'local2_HCPtest_12_{run.replace("_", "")}_homo2_0.045_{hemi}.csv'),
        Local_homo, delimiter=',', fmt='%.6f'
    )
    np.savetxt(
        os.path.join(save_path, f'Orig2_HCPtest_12_{run.replace("_", "")}_homo2_0.045_{hemi}.csv'),
        Orig_homo, delimiter=',', fmt='%.6f'
    )


def main():
    """Main function to process all subjects."""
    run = "R1_LR"
    print(run)

    network_label_path = '/mnt/data0/home/qwang/1000subjects_reference/1000subjects_clusters017_ref.mat'
    work_path = '/mnt/data0/home/qwang/DATA/HCP/'

    for hemi in ['lh', 'rh']:
        temp_label = load_network_labels(network_label_path, hemi)
        temp_local = pd.read_csv(f"{work_path}/Temp_homo2_{hemi}.csv", header=None).values.flatten()
        
        ad = pd.read_csv(f"/mnt/data0/home/qwang/DATA/HCP/fs5_secondadjacent_{hemi}.csv", header=None).values
        n, m = ad.shape

        sub_name = pd.read_csv(os.path.join(work_path, "HCP_test.csv"), header=None)
        ls = sub_name.values

        for subject in ls[:, 0]:
            process_subject_data(work_path, str(int(subject)), hemi, temp_label, temp_local, ad, n, m, run)

if __name__ == "__main__":
    main()