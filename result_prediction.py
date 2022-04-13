# -*- coding: utf-8 -*-
"""
Created on 2022-04-06
Author ZhengRui
Co-author LongJianghua
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')
import sys
import json
import traceback
import joblib

class Result:
    precision = 0
    recall = 0
    accuracy = 0
    rocX = []
    rocY = []
    featureImportances = []

params = {}
params['model'] = '../model/paderborn.model'
params['test'] = '../datafeature/testdataB_feature_select.csv'
params['opath'] = '../result/result.csv'

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

    #导入模型
    model = joblib.load(params['model'])

    #导入测试集数据
    test_csv = pd.read_csv(params['test'])
    test_feature = test_csv
    
    data = np.array(test_feature)
    number = data.shape[0]
    
    #数据归一化处理
    test_feature = preprocessing.MinMaxScaler().fit_transform(test_feature)
    
    #预测概率
    predict = model.predict_proba(test_feature)
    #预测结果
    predict = model.predict(test_feature)

    #导出结果至数组Predict
    Predict = [i for i in range(number)]
    for i in range(number):
        Predict[i] = int(predict[i])
    
    #导出结果至csv
    result = pd.DataFrame(Predict)
    result.to_csv(params['opath'], index=False)
    
    
except Exception as e:
    traceback.print_exc()
    print(e)