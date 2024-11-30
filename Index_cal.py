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

# Inputs
run="R1_LR"
print(run)
for hemi in ['lh','rh']:
    Network_Label=scio.loadmat('/mnt/data0/home/qwang/'+'1000subjects_reference/1000subjects_clusters017_ref.mat')
    Temp_Label=Network_Label[hemi+'_labels']
    Temp_Label=Temp_Label.flatten()
    work_path='/mnt/data0/home/qwang/DATA/HCP/'   
    
    name=pd.read_csv("/mnt/data0/home/qwang/DATA/HCP/fs5_secondadjacent_"+hemi+".csv",header=None)
    ad=name.values
    n,m=ad.shape
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


    # Yeo Template
    Temp_local=pd.read_csv("/mnt/data0/home/qwang/DATA/HCP/Temp_homo2_"+hemi+".csv",header=None).values
    # data
    sub_name=pd.read_csv(work_path+"HCP_test.csv",header=None)
    ls=sub_name.values
   
    for subject in ls[:,0]:
        subject=str(int(subject))
       
        save_path=work_path+"Output/"+subject+"/"
        label=pd.read_csv(save_path+"final_iter_label_0.045_pial_"+hemi+".csv",header=None).values
        Local_homo=np.zeros(n)
        Orig_homo=np.zeros(n)
        for i in range(n):
            if Temp_Label[i]==0:
                Local_homo[i]=0
                Orig_homo[i]=0
                continue
            adjacent=label[ad[i]-1]
            # Get unique elements and their counts
            unq,counts = np.unique(adjacent,return_counts=True)
            temp=0
            N=m
            for j in range(unq.shape[0]):
                if unq[j]==0:
                    N=m-counts[j]
                    continue    
                temp+=math.pow(counts[j],2)
            t=math.sqrt(temp)/N
            # Local_homo[i]=(t-math.sqrt(23)/19)/(1-math.sqrt(23)/19)-Temp_local[i]
            # Orig_homo[i]=(t-math.sqrt(23)/19)/(1-math.sqrt(23)/19)
            # Local_homo[i]=Temp_local[i]-(t-math.sqrt(23)/19)/(1-math.sqrt(23)/19)
            # 1
            # Orig_homo[i]=1-(t-math.sqrt(7)/7)/(1-math.sqrt(7)/7)
            # Local_homo[i]=Orig_homo[i]-Temp_local[i]
            # 2
            Orig_homo[i]=1-(t-math.sqrt(23)/19)/(1-math.sqrt(23)/19)
            Local_homo[i]=Orig_homo[i]-Temp_local[i]
            # 3
            # Local_homo[i]=(t-math.sqrt(47)/27)/(1-math.sqrt(47)/27)-Temp_local[i]
            # Orig_homo[i]=(t-math.sqrt(47)/27)/(1-math.sqrt(47)/27)
            # Orig_homo[i]=1-(t-math.sqrt(47)/27)/(1-math.sqrt(47)/27)
            # Local_homo[i]=Orig_homo[i]-Temp_local[i]

        np.savetxt(save_path+'local2_HCPtest_12_'+run.replace('_','')+'_homo2_0.045_'+hemi+'.csv',Local_homo,delimiter=',',fmt='%.6f')
        np.savetxt(save_path+'Orig2_HCPtest_12_'+run.replace('_','')+'_homo2_0.045_'+hemi+'.csv',Orig_homo,delimiter=',',fmt='%.6f')




