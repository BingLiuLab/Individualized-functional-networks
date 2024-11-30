"""
# @ creater by Qi Wang on 2023.2.16
# Description: Calculate delta LNC and network scale LNC values.
# Usage:
    A directory of database data results needs to be established and the location of the data to be matched should be 
    determined with the data/subject name as the unique index.
"""

import scipy.io as scio
import numpy as np
import random
import csv
import os
import pandas as pd
import math
# import pingouin as pg

os.environ["OUTDATED_IGNORE"] = "1"

def similarity(v_a,v_b):
    num=0
    for i in range(v_a.shape[0]):
        if v_a[i]==v_b[i]:
            num+=1
    return num/v_a.shape[0]

draw_path="/mnt/data0/home/qwang/DATA/HCP/Output/"
work_path='/mnt/data0/home/qwang/DATA/HCP/'


str1="12"
# for ss in ["R1_LR","R1_RL","R2_LR","R2_RL"]:
for ss in ["R1_LR"]:
    Type='HCPtest_12_'+ss.replace('_','')
    print(Type)
    name=pd.read_csv(work_path+"HCP_test.csv",header=None)
   
    ls=name.values
    ls_base=ls[:,0]
    n=ls_base.shape[0]

    # # 17network statistics 
    Network_Label=scio.loadmat('/mnt/data0/home/qwang/'+'1000subjects_reference/1000subjects_clusters017_ref.mat')
    Temp_Label_lh=Network_Label['lh_labels']
    Temp_Label_rh=Network_Label['rh_labels']
    Temp_Label_lh=Temp_Label_lh.flatten()
    Temp_Label_rh=Temp_Label_rh.flatten()
    Temp_Label=np.hstack((Temp_Label_lh,Temp_Label_rh))
    # Temp_entro=pd.read_csv("/mnt/data0/home/qwang/DATA/HCP/Template_17network_entropy.csv",header=None).values

    s="_0.045_"
    Res=[]
    Res_del=[]
    Res_del_abs=[]
    Ent=[]
    Ent_del=[]
    homo_v=[]
    homo_v_delta=[]

    for subject in ls_base:
        Stats_homo=np.zeros(17)
        Stats_homo_delta=np.zeros(17)
        Stats_homo_abs_delta=np.zeros(17)
        num_homo=np.zeros(17)
        Res_homo=np.zeros(17)
        Res_homo_delta=np.zeros(17)
        Res_homo_delta_abs=np.zeros(17)
        Entro=np.zeros(17)
        Entro_delta=np.zeros(17)

        subject=str(int(subject))
    
        save_path=work_path+"Output/"+subject+"/"

        # input file
        Local_lh=pd.read_csv(save_path+"Orig2_"+Type+"_homo2"+s+"lh.csv",header=None).values
        Local_rh=pd.read_csv(save_path+"Orig2_"+Type+"_homo2"+s+"rh.csv",header=None).values
        Local_lh=np.array(Local_lh)
        Local_rh=np.array(Local_rh)
        Local=np.vstack((Local_lh,Local_rh))
        homo_v.append(Local.T.flatten())
        n=Local.shape[0]

        Local_delta_lh=pd.read_csv(save_path+"local2_"+Type+"_homo2"+s+"lh.csv",header=None).values
        Local_delta_rh=pd.read_csv(save_path+"local2_"+Type+"_homo2"+s+"rh.csv",header=None).values
        Local_delta_lh=np.array(Local_delta_lh)
        Local_delta_rh=np.array(Local_delta_rh)
        Local_delta=np.vstack((Local_delta_lh,Local_delta_rh))
        
        homo_v_delta.append(Local_delta.T.flatten())


        label_lh=pd.read_csv(save_path+"final_iter_label"+s+"pial_lh.csv",header=None).values
        label_rh=pd.read_csv(save_path+"final_iter_label"+s+"pial_rh.csv",header=None).values
        label_lh=np.array(label_lh)
        label_rh=np.array(label_rh)
        label=np.vstack((label_lh,label_rh))

        for i in range(n):
            if label[i]==0:
                continue
        
            t=int(label[i]-1)
            # t=int(Temp_Label[i]-1)
            
            Stats_homo[t]+=Local[i]
            Stats_homo_delta[t]+=Local_delta[i]
            Stats_homo_abs_delta[t]+=abs(Local_delta[i])
            num_homo[t]+=1
    
        
        for j in range(17):
            Res_homo[j]=Stats_homo[j]/num_homo[j]
            Res_homo_delta[j]=Stats_homo_delta[j]/num_homo[j]
            Res_homo_delta_abs[j]=Stats_homo_abs_delta[j]/num_homo[j]
            e_t=num_homo[j]/n
            if e_t==0:
                Entro[j]=0
            Entro[j]=0-e_t*np.log2(e_t)
            Entro_delta[j]=0-e_t*np.log2(e_t)-Temp_entro[j]

        Res.append(Res_homo)
        Res_del.append(Res_homo_delta)
        Res_del_abs.append(Res_homo_delta_abs)
        Ent.append(Entro)
        Ent_del.append(Entro_delta)

    np.savetxt(work_path+Type+'_homo_vertex2'+s+'_20.csv',np.array(homo_v),delimiter=',',fmt='%.6f')
    np.savetxt(work_path+Type+'_homo_vertex_delta2'+s+'_20.csv',np.array(homo_v_delta),delimiter=',',fmt='%.6f')
    np.savetxt(work_path+Type+'_homo2'+s+'_20.csv',np.array(Res),delimiter=',',fmt='%.6f')
    np.savetxt(work_path+Type+'_homo_delta2'+s+'_20.csv',np.array(Res_del),delimiter=',',fmt='%.6f')

    print("OK")