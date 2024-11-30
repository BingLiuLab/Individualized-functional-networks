"""
# @ creater by Qi Wang on 2022.8.6
# Description: Individualized calculation function interface for data calls. 
  The data in the matching library and the data to be matched are both called from this. 
  For specific function parameters, please refer to Muti_runs-Parcellate.py.
# Usage:
    By using a unique index of data/subject names, as the matching database data contains multiple runs 
    of data, even if the data to be matched is single run data, a run name needs to be set.
"""
# import nilearn
import scipy.io as scio
import numpy as np
import random
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import math
import Muti_runs_Parcellate as MP
import time
# os.environ["OUTDATED_RAISE_EXCEPTION"] = "1"
os.environ["OUTDATED_IGNORE"] = "1"



work_path='/mnt/data0/home/qwang/DATA/GSP/'
name=pd.read_csv(work_path+"HCP_behavior.csv")
# name=pd.read_csv(work_path+"HCP_test.csv",header=None)
ls=name.values
# session=["R1_LR","R1_RL","R2_RL","R2_LR"] # Muti-runs
# ss1=session[1]

session=["Run2"]
# print(session)
Network_Label=scio.loadmat('/mnt/data0/home/qwang/1000subjects_reference/1000subjects_clusters017_ref.mat')
Label_lh=Network_Label['lh_labels']
Label_rh=Network_Label['rh_labels']
Label_lh=Label_lh.flatten()
Label_rh=Label_rh.flatten()

# First adjacent matrix for each vertex
Ad_lh=pd.read_csv("/mnt/data0/home/qwang/DATA/HCP/fs5_firstadjacent_lh.csv",header=None).values 
Ad_rh=pd.read_csv("/mnt/data0/home/qwang/DATA/HCP/fs5_firstadjacent_rh.csv",header=None).values
for i,ss in enumerate(session):  
    ss0=[]
    ss0.append(ss)
    print(ss0)
    for subject in ls[:,0]:
  
        # time_start = time.time() 
        # subject=str(int(subject))
        MP.Indivdual_parcellate(subject=subject,output_path=work_path+'Output/'+ss0[0]+"/",work_path=work_path+'Output/',ss=session,L_lh=Label_lh,L_rh=Label_rh,Adjacent_lh=Ad_lh,Adjacent_rh=Ad_rh,Single=True)
        # time_end = time.time()  
        # time_sum = time_end - time_start  
        # print(time_sum)