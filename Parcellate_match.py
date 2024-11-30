"""
# @ creater by Qi Wang on 2022.11.21
# Description: For general matching data, prior calculations are performed first, 
  and the result with the highest similarity in the library is used as the initial space based on prior 
  similarity. The number of secondary iterations is determined based on similarity rather than convergence.
# Usage:
    A directory of database data results needs to be established and the location of the data to be matched should be 
    determined with the data/subject name as the unique index.
"""

import scipy.io as scio
import numpy as np
# import csv
import os
import pandas as pd
import math
import Muti_runs_Parcellate as MP
# from dtaidistance import dtw
import time


def mean2(x):
    y = np.sum(x) / np.size(x)
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum())
    return r

def similarity(v_a,v_b):
    num=0
    for i in range(v_a.shape[0]):
        if v_a[i]==v_b[i]:
            num+=1
    return num/v_a.shape[0]

manhattan_distance = lambda x, y: np.abs(x - y)


draw_path="/mnt/data0/home/qwang/DATA/HCP/Output/"
work_path='/mnt/data0/home/qwang/DATA/GSP/'
mid=[]

# Batch draw
data_base=pd.read_csv("/mnt/data0/home/qwang/DATA/HCP/HCP_behavior.csv")
data_test=pd.read_csv("/mnt/data0/home/qwang/DATA/GSP/GSP_behavior.csv",header=None)
# print(data_base.shape)
ls1=data_base.values
ls2=data_test.values
n=969
flag=0
ls_base=ls1[:n,0]
# n=ls1.shape[0]

ls_test=ls2[:,0]
# Geodesic distance
Ad_lh=pd.read_csv("/mnt/data0/home/qwang/DATA/HCP/fs5_firstadjacent_lh.csv",header=None).values 
Ad_rh=pd.read_csv("/mnt/data0/home/qwang/DATA/HCP/fs5_firstadjacent_rh.csv",header=None).values

Ad=np.vstack((Ad_lh,Ad_rh))
ad_n=Ad.shape[0]
ss=["R1_LR","R1_RL","R2_RL","R2_LR"]
ss_base=["Run2"]

print(ss_base)
# Time_dict = np.load(work_path+'Train_Time.npy', allow_pickle='TRUE')[()]

# # Yeo 17 network
Network_Label=scio.loadmat('/mnt/data0/home/qwang/1000subjects_reference/1000subjects_clusters017_ref.mat')
Label_match_lh=Network_Label['lh_labels']
Label_match_rh=Network_Label['rh_labels']
# Label_match_lh=Label_match_lh.flatten()
# Label_match_rh=Label_match_rh.flatten()
Label_match=np.vstack((Label_match_lh,Label_match_rh)).flatten()

ver_sel=np.where(Label_match.flatten()!=0)
idx_len=len(ver_sel[0])


Label_matrix=np.zeros((n,idx_len))
# Prior calculation
for p in range(n):
    subject=str(int(ls_base[p]))
    Label_base_lh=pd.read_csv(draw_path+subject+"/"+'final_iter_label_0.045_pial_lh.csv',header=None,dtype=int).values
    Label_base_rh=pd.read_csv(draw_path+subject+"/"+'final_iter_label_0.045_pial_rh.csv',header=None,dtype=int).values
    Label_base=np.vstack((Label_base_lh,Label_base_rh))
   
    Label_base=Label_base[ver_sel]
    Label_matrix[p]=Label_base.flatten()

   
num_iter=[]
num_cor=[]

for sub in ls_test:

    Match_res=[]
    cor_max=-1
    idx_max=0
    print(sub)
    # sub=str(int(sub))
    Seed=[]

    # Time_test_lh=pd.read_csv(work_path+ss_base[0]+"/"+sub+"/surf_lh_time_pial.csv",header=None).values # 10242*1200 timecourse
    # Time_test_rh=pd.read_csv(work_path+ss_base[0]+"/"+sub+"/surf_rh_time_pial.csv",header=None).values 
    # # n1,m1=Time_test_lh.shape


    Label_test_lh=pd.read_csv(work_path+"Output/"+ss_base[0]+"/"+sub+'/'+ss_base[0]+'535final_iter_label_0.045_pial_lh.csv',header=None,dtype=int).values
    Label_test_rh=pd.read_csv(work_path+"Output/"+ss_base[0]+'/'+sub+"/"+ss_base[0]+'535final_iter_label_0.045_pial_rh.csv',header=None,dtype=int).values
    Label_test=np.vstack((Label_test_lh,Label_test_rh))
    Label_test=Label_test.flatten()
   
    Label_test=Label_test[ver_sel]
    
# Match
    for i in range(n):
        
        subject=str(int(ls_base[i]))
        Match_cor=similarity(Label_matrix[i],Label_test)

        temp=[int(subject),Match_cor]
    #     # 
        if Match_cor>cor_max:
            cor_max=Match_cor
            idx_max=i
        Match_res.append(temp)
  
   
    idx=str(int(ls_base[idx_max]))
    Label_lh=pd.read_csv(draw_path+idx+"/"+'final_iter_label_0.045_pial_lh.csv',header=None,dtype=int).values
    Label_rh=pd.read_csv(draw_path+idx+"/"+'final_iter_label_0.045_pial_rh.csv',header=None,dtype=int).values
    # # Define iters according to the cor2
    alpha=0.025
    num=round((0.999-cor_max)/alpha)+1
    
   

    # Match_res=np.nan_to_num(np.array(Match_res),nan=0)
    # np.savetxt(draw_path+'HCP_test/'+ss_base[0]+'/'+sub+'/Yeo31_Match_res.csv',Match1,delimiter=',',fmt='%f')  
    # np.savetxt(work_path+'Output/Match/'+ss_base[0]+'/'+sub+'/GSP_Run1_no5_Match_res.csv',np.array(Match_res),delimiter=',',fmt='%f')  
    MP.Indivdual_parcellate(subject=sub,output_path=work_path+'Output/Match/'+ss_base[0]+'/',work_path=work_path+'Output/',ss=ss_base,L_lh=Label_lh,L_rh=Label_rh,Adjacent_lh=Ad_lh,Adjacent_rh=Ad_rh,num_Iter=num,Single=True)
  