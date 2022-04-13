# -*- coding: utf-8 -*-
"""
Created on 2022-04-06
Author ZhengRui
Co-author LongJianghua
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold #交叉验证
from sklearn.model_selection import GridSearchCV #网格搜索
from sklearn.model_selection import train_test_split #将数据集分开成训练集和测试集
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot
from sklearn.metrics import r2_score
import sklearn.utils as uts  # 打乱数据
import warnings
warnings.filterwarnings('ignore')
import sys
import json
import traceback
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score

class Result:
    precision = 0
    recall = 0
    accuracy = 0
    rocX = []
    rocY = []
    featureImportances = []
params = {}

#调参
params['n_estimators'] = 240 #1000
params['max_depth'] = 16
params['min_samples_leaf'] = 10
params['min_samples_split'] = 15
params['max_features'] = 2

params['train'] = '../datasmoted/traindata_N15_M01_F10_feature_select_smoted.csv'
params['test'] = '../datafeature/traindata_N15_M07_F04_feature_select.csv'

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

    #训练集
    train = np.array(pd.read_csv(params['train']))
    train_y = train[:, -1]
    train_x = train[:, :-1]
    train_x, train_y = uts.shuffle(train_x, train_y, random_state=12)  # 打乱样本

    #测试集
    test = np.array(pd.read_csv(params['test']))
    test_y = test[:, -1]
    test_x = test[:, :-1]
    
    #数据归一化
    train_x = preprocessing.MinMaxScaler().fit_transform(train_x)
    test_x = preprocessing.MinMaxScaler().fit_transform(test_x)
    
    clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                max_features=params['max_features'],
                                max_depth=params['max_depth'],
                                min_samples_split=params['min_samples_split'],
                                min_samples_leaf=params['min_samples_leaf'],
                                random_state=10,
                                oob_score=True).fit(train_x, train_y)
    
    #获得对应评价指标
    predict = clf.predict(test_x)
    precision = precision_score(test_y, predict, average='macro')
    recall = recall_score(test_y, predict, average='macro')
    accuracy = accuracy_score(test_y, predict)
    
    #特征重要度
    features = list(pd.read_csv(params['test']).columns)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    num_features = len(importances)
    
    #将特征重要度以柱状图展示
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(num_features), importances[indices], color="g", align="center")
    plt.xticks(range(num_features), [features[i] for i in indices], rotation='45')
    plt.xlim([-1, num_features])
    plt.show()
    
    '''
    #输出各个特征的重要度
    for i in indices:
        print ("{0} - {1:.3f}".format(features[i], importances[i]))

    '''
    
    #输出模型参数
    print('n_estimators=',clf.n_estimators)
    print('max_depth=',clf.max_depth)
    print('min_samples_leaf=',clf.min_samples_leaf)
    print('min_samples_split=',clf.min_samples_split)
    print('max_features=',clf.max_features)

    res = {}
    res['precision'] = precision
    res['recall'] = recall
    res['accuracy'] = accuracy
    res['fMeasure'] = f1_score(test_y, predict, average='macro')
    
    print(json.dumps(res))
    
    joblib.dump(clf,'../model/paderborn.model')
    print('袋外验证分数：')
    print(clf.oob_score_)
    
    
except Exception as e:
    traceback.print_exc()
    print(e)
















