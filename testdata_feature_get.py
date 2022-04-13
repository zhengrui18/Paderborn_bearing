# -*- coding: utf-8 -*-
"""
Created on 2022-04-06
Author ZhengRui
Co-author LongJianghua
"""

#用于特征提取
from cProfile import label
from operator import length_hint
import pandas as pd
import numpy as np
import os
from scipy import stats,signal,fftpack
import math
from pywt import wavedec
import traceback
import sys

params = {}

# testdataA
# params['testApath'] = '../dataset/testdataA.csv'
# params['testAopath'] = '../datafeature/testdataA_feature_select.csv'

#testdataB
params['testBpath'] = '../dataset/testdataB.csv'
params['testBopath'] = '../datafeature/testdataB_feature_select.csv'

argvs = sys.argv
try:
    for i in range(len(argvs)):
        if i < 1:
            continue
        if argvs[i].split('=')[1] == 'None':
            params[argvs[i].split('=')[0]] = None
        else:
            Type = type(params[argvs[i].split('=')[0]])
            params[argvs[i].split('=')[0]] = Type(argvs[i].split('=')[1])
            
    # columns_list_all = ['time_mean','time_std','time_max','time_min','time_rms', \
    #                     'time_ptp','time_median','time_iqr','time_pr','time_sknew', \
    #                     'time_kurtosis','time_var','time_amp','time_smr','time_wavefactor', \
    #                     'time_peakfactor','time_pulse','time_margin', \
    #                     'freq_mean','freq_std','freq_max','freq_min','freq_rms','freq_median', \
    #                     'freq_iqr','freq_pr','freq_f2','freq_f3','freq_f4', \
    #                     'freq_f5','freq_f6','freq_f7','freq_f8','ener_cA5', \
    #                     'ener_cD1','ener_cD2','ener_cD3','ener_cD4','ener_cD5', \
    #                     'ratio_cA5','ratio_cD1','ratio_cD2','ratio_cD3','ratio_cD4','ratio_cD5']
    
    
    #15 features
    #columns_list_all = ['time_mean',   'time_std',     'time_median',\
    #                    'time_iqr',     'time_pr',      'time_amp',\
    #                   'freq_max',     'freq_median',  'freq_iqr',\
    #                    'freq_f5',      'freq_f6',      'freq_f7', \
    #                   'freq_f8',      'ener_cD1',     'ratio_cD1']

    #feature selected
    columns_list_all = ['time_mean',   'time_median',  'freq_mean',\
                        'freq_std',     'freq_median',  'freq_f2', \
                        'freq_f5',      'freq_f6',      'freq_f7', \
                        'freq_f8',      'ener_cD1',     'ratio_cD1']
                        
    #记录各部分的特征数目
    #div = {'time':18, 'freq':15,'ener_ratio':12}
    
    def feature_get(filepath):
        
        dfs = pd.read_csv(filepath)
        col_1 = list(dfs.columns)
        v = col_1[:-1]
        df = dfs.loc[:,v]
        data = np.array(df[v])
        length_get = np.array(dfs['2558'])
        
        feature_list = [i for i in range(len(length_get))]
        
        for i in range(data.shape[0]):
            df_line = data[i,:]
            
            #time-domain feature,18
            #时域特征 18个

            time_mean = df_line.mean()
            time_std = df_line.std()
            # time_max = df_line.max()
            # time_min = df_line.min()
            # time_rms = np.sqrt(np.square(df_line).mean())
            # time_ptp = time_max-time_min 
            time_median = np.median(df_line)
            time_iqr = np.percentile(df_line,75)-np.percentile(df_line,25)
            time_pr = np.percentile(df_line,90)-np.percentile(df_line,10)
            # time_skew = stats.skew(df_line)
            # time_kurtosis = stats.kurtosis(df_line)
            # time_var = np.var(df_line)
            time_amp = np.abs(df_line).mean()
            # time_smr = np.square(np.sqrt(np.abs(df_line)).mean())

            #下面四个特征需要注意分母为0或接近0问题，可能会发生报错
            # time_wavefactor = time_rms/time_amp
            # time_peakfactor = time_max/time_rms
            # time_pulse = time_max/time_amp
            # time_margin = time_max/time_smr
            

            #freq-domain feature,15
            #频域特征 15个

            #采样频率25600Hz
            df_fftline = fftpack.fft(df_line)
            freq_fftline = fftpack.fftfreq(len(df_line),1/25600)
            df_fftline = abs(df_fftline[freq_fftline>=0])
            freq_fftline = freq_fftline[freq_fftline>=0]

            #基本特征,依次为均值，标准差，最大值，最小值，均方根，中位数，四分位差，百分位差
            freq_mean = df_fftline.mean()
            freq_std = df_fftline.std()
            freq_max = df_fftline.max()
            # freq_min = df_fftline.min()
            # freq_rms = np.sqrt(np.square(df_fftline).mean())
            freq_median = np.median(df_fftline)
            freq_iqr = np.percentile(df_fftline,75)-np.percentile(df_fftline,25)
            # freq_pr = np.percentile(df_fftline,90)-np.percentile(df_fftline,10)

            #f2 f3 f4反映频谱集中程度
            freq_f2 = np.square((df_fftline-freq_mean)).sum()/(len(df_fftline)-1)
            # freq_f3 = pow((df_fftline-freq_mean),3).sum()/(len(df_fftline)*pow(freq_f2,1.5))
            # freq_f4 = pow((df_fftline-freq_mean),4).sum()/(len(df_fftline)*pow(freq_f2,2))

            #f5 f6 f7 f8反映主频带位置
            freq_f5 = np.multiply(freq_fftline,df_fftline).sum()/df_fftline.sum()
            freq_f6 = np.sqrt(np.multiply(np.square(freq_fftline),df_fftline).sum())/df_fftline.sum()
            freq_f7 = np.sqrt(np.multiply(pow(freq_fftline,4),df_fftline).sum())/np.multiply(np.square(freq_fftline),df_fftline).sum()
            freq_f8 = np.multiply(np.square(freq_fftline),df_fftline).sum()/np.sqrt(np.multiply(pow(freq_fftline,4),df_fftline).sum()*df_fftline.sum())
            

            #----------  timefreq-domain feature,12
            #时频域特征 12个

            # 5级小波变换，最后输出6个能量特征和其归一化能量特征
            cA5, cD5, cD4, cD3, cD2, cD1 = wavedec(df_line, 'db10', level=5)
            ener_cA5 = np.square(cA5).sum()
            ener_cD5 = np.square(cD5).sum()
            ener_cD4 = np.square(cD4).sum()
            ener_cD3 = np.square(cD3).sum()
            ener_cD2 = np.square(cD2).sum()
            ener_cD1 = np.square(cD1).sum()
            ener = ener_cA5 + ener_cD1 + ener_cD2 + ener_cD3 + ener_cD4 + ener_cD5
            # ratio_cA5 = ener_cA5/ener
            # ratio_cD5 = ener_cD5/ener
            # ratio_cD4 = ener_cD4/ener
            # ratio_cD3 = ener_cD3/ener
            # ratio_cD2 = ener_cD2/ener
            ratio_cD1 = ener_cD1/ener


            # feature_list[i]=[time_mean,time_std,time_max,time_min,time_rms,time_ptp,time_median,time_iqr,time_pr,time_skew,time_kurtosis,time_var,time_amp,
            #                   time_smr,time_wavefactor,time_peakfactor,time_pulse,time_margin,freq_mean,freq_std,freq_max,freq_min,freq_rms,freq_median,
            #                   freq_iqr,freq_pr,freq_f2,freq_f3,freq_f4,freq_f5,freq_f6,freq_f7,freq_f8,ener_cA5,ener_cD1,ener_cD2,ener_cD3,ener_cD4,ener_cD5,
            #                   ratio_cA5,ratio_cD1,ratio_cD2,ratio_cD3,ratio_cD4,ratio_cD5,]
            
            # feature_list[i] = [ time_mean,   time_std,     time_median,\
            #                     time_iqr,     time_pr,      time_amp,\
            #                     freq_max,     freq_median,  freq_iqr,\
            #                     freq_f5,      freq_f6,      freq_f7, \
            #                     freq_f8,      ener_cD1,     ratio_cD1]
            
            feature_list[i] = [ time_mean,  time_median,    freq_mean, \
                                freq_std,   freq_median,    freq_f2, \
                                freq_f5,    freq_f6,        freq_f7, \
                                freq_f8,    ener_cD1,       ratio_cD1]
            
            
            
        print('Feature_get is finished!')
        return feature_list
    
    
    #测试集A特征提取
    # file_path = params['testApath']
    # features = feature_get(file_path)
    # col_lab = columns_list_all
    # result = pd.DataFrame(features, columns = col_lab)
    # result.to_csv(params['testAopath'], sep=',', header=True, index=False)

    #测试集B特征提取
    file_path = params['testBpath']
    features = feature_get(file_path)
    col_lab = columns_list_all
    result = pd.DataFrame(features, columns = col_lab)
    result.to_csv(params['testBopath'], sep=',', header=True, index=False)
        
except Exception as e:
    traceback.print_exc()
    print(e)