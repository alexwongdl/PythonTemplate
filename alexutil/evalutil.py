"""
Created by Alex Wang on 2018-01-09
"""
import numpy as np
from sklearn import metrics

def cal_precision_recall(predict, tags):
    """
    计算roc
    :param predict: (batch_size, 2)
    :param tags:(batch_size, 1) 0/1
    :return:
    print('roc thresholds:{}'.format(','.join(['{:.4f}'.format(item) for item in roc_thresholds])))
    print('           tpr:{}'.format(','.join(['{:.4f}'.format(item) for item in tpr])))
    print('           fpr:{}'.format(','.join(['{:.4f}'.format(item) for item in fpr])))
    print('')
    print('pr thresholds:{}'.format(','.join(['{:.4f}'.format(item) for item in pr_thresholds])))
    print('         prec:{}'.format(','.join(['{:.4f}'.format(item) for item in prec_list])))
    print('       recall:{}'.format(','.join(['{:.4f}'.format(item) for item in recall_list])))
    """
    elapsion = 1e-8

    TP, FP, FN = 0, 0, 0
    predict_idx = np.argmax(predict, axis=1)
    for i in range(len(tags)):
        if tags[i] == 1 and predict_idx[i] == 1:
            TP += 1
        if tags[i] == 0 and predict_idx[i] == 1:
            FP += 1
        if tags[i] == 1 and predict_idx[i] == 0:
            FN += 1

    precision = TP / (TP + FP + elapsion)
    recall = TP / (TP + FN + elapsion)

    # calculate auc
    pred = predict[:, 1]
    fpr, tpr, roc_thresholds = metrics.roc_curve(tags, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    prec_list, recall_list, pr_thresholds = metrics.precision_recall_curve(tags, pred, pos_label=1)

    return precision, recall, tags, predict_idx, auc, fpr, tpr, roc_thresholds, prec_list, recall_list, pr_thresholds

