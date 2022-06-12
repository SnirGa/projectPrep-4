import statistics
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

#import ann
import knn
import random_forest
import svm
from knn import get_metrics
from naive_bayes import get_gnb_results

sns.set()




def get_statistics_results(model,csv):
    if model == 'nb':
        accuracy_lst, tpr_lst, fpr_lst, precision_lst, auc_roc_curve_lst, auc_precision_recall_lst, training_time_lst, inference_time_lst = get_gnb_results(csv)
    elif model == 'knn':
        accuracy_lst, tpr_lst, fpr_lst, precision_lst, auc_roc_curve_lst, auc_precision_recall_lst, training_time_lst, inference_time_lst = knn.get_metrics_updated(csv)
    elif model=='rf':
        accuracy_lst, tpr_lst, fpr_lst, precision_lst, auc_roc_curve_lst, auc_precision_recall_lst, training_time_lst, inference_time_lst = random_forest.get_rf_results(
            csv)
    elif model=='ann':
        accuracy_lst, tpr_lst, fpr_lst, precision_lst, auc_roc_curve_lst, auc_precision_recall_lst, training_time_lst, inference_time_lst = ann.get_ann_results(csv)
    elif model=='svm':
        accuracy_lst, tpr_lst, fpr_lst, precision_lst, auc_roc_curve_lst, auc_precision_recall_lst, training_time_lst, inference_time_lst = svm.get_svm_results(
            csv)

    mean_row = [model,statistics.mean(accuracy_lst), statistics.mean(tpr_lst), statistics.mean(fpr_lst),
                statistics.mean(precision_lst), statistics.mean(auc_roc_curve_lst),
                statistics.mean(auc_precision_recall_lst), statistics.mean(training_time_lst),
                statistics.mean(inference_time_lst)]
    std_row = [model,statistics.stdev(accuracy_lst), statistics.stdev(tpr_lst), statistics.stdev(fpr_lst),
               statistics.stdev(precision_lst), statistics.stdev(auc_roc_curve_lst),
               statistics.stdev(auc_precision_recall_lst), statistics.stdev(training_time_lst),
               statistics.stdev(inference_time_lst)]
    return mean_row, std_row



def get_graph(flag,csv):
    #flag- mean | std
    #TODO: change the parameters to the right ones
    nb_mean_row, nb_std_row = get_statistics_results('nb',csv)
    knn_mean_row,knn_std_row=get_statistics_results('knn',csv)
    rf_mean_row,rf_std_row=get_statistics_results('rf',csv)
    ann_mean_row,ann_std_row=get_statistics_results('ann',csv)
    svm_mean_row,svm_std_row=get_statistics_results('svm',csv)

    knn_mean_row[0]='knn'
    rf_mean_row[0]='rf'
    ann_mean_row[0]='ann'
    svm_mean_row[0]='svm'
    if flag=='mean':
        merged = [nb_mean_row,knn_mean_row,rf_mean_row,ann_mean_row,svm_mean_row]
    else:
        merged = [nb_std_row,knn_std_row,rf_std_row,ann_std_row,svm_std_row]
        #accuracy_lst, tpr_lst, fpr_lst, precision_lst, auc_roc_curve_lst, auc_precision_recall_lst, training_time_lst, inference_time_lst = get_gnb_results()
    df=pd.DataFrame(merged,columns=['model','accuracy','tpr','fpr','precision','auc_roc_curve','auc_precision_recall','training_time','inference_time'])
    #ax=sns.barplot(x='model',y=["accuracy","tpr"],data=df)
    ax=df.plot(x="model", y=['accuracy','tpr','fpr','precision','auc_roc_curve','auc_precision_recall','training_time','inference_time'], kind="bar")
    ax.legend(bbox_to_anchor=(1.1, 1.05))
    plt.show()
#get_graph('mean','all_datasets/breast-cancer.csv')

# nb_mean_row, nb_std_row = get_statistics_results('nb','datasets/breast-cancer.csv')
# knn_mean_row,knn_std_row=get_statistics_results('knn','datasets/breast-cancer.csv')
# rf_mean_row,rf_std_row=get_statistics_results('rf','datasets/breast-cancer.csv')
# ann_mean_row,ann_std_row=get_statistics_results('ann','datasets/breast-cancer.csv')
# svm_mean_row,svm_std_row=get_statistics_results('svm','datasets/breast-cancer.csv')
# try_dict={
#     'method':['model','accuracy','tpr','fpr','precision','auc_roc_curve','auc_precision_recall','training_time','inference_time'],
#     'nb':nb_mean_row,
#     'knn':knn_mean_row,
#     'rf':rf_mean_row,
#     #'ann':ann_mean_row,
#     'svm':svm_mean_row
# }
# df2=pd.DataFrame(try_dict,index=['model','accuracy','tpr','fpr','precision','auc_roc_curve','auc_precision_recall','training_time','inference_time'])
#
#
# df2=df2[1:]
# ax = df2[:-2].plot(x="method",
#              y=['nb', 'knn', 'rf', 'ann', 'svm'], kind="bar")
# #ax= df2[len(df2)-2:len(df2)].plot(x="method",
# #             y=['nb', 'knn', 'rf', 'ann', 'svm'], kind="bar")
# plt.xticks(rotation=360)
#
# plt.show()
# print(df2)
def single_graph(model,csv,operation,graph_title,num):
    model_mean_row, model_std_row = get_statistics_results(model, csv)
    if operation=='mean':
        row=model_mean_row
    else:
        row=model_std_row
    try_dict = {
        'metrics': ['model', 'accuracy', 'tpr', 'fpr', 'precision', 'auc_roc_curve', 'auc_precision_recall',
                   'training_time', 'inference_time'],
        'values':row
    }
    df2 = pd.DataFrame(try_dict,
                       index=['model', 'accuracy', 'tpr', 'fpr', 'precision', 'auc_roc_curve', 'auc_precision_recall',
                              'training_time', 'inference_time'])

    df2 = df2[1:]
    if num==1:
        ax = df2[:-2].plot(x="metrics",y='values', kind="bar")
    else:
        ax = df2[len(df2) - 2:len(df2)].plot(x="metrics",y='values', kind="bar")


    # ax= df2[len(df2)-2:len(df2)].plot(x="method",
    #             y=['nb', 'knn', 'rf', 'ann', 'svm'], kind="bar")
    plt.xticks(rotation=360)
    plt.title(graph_title,loc='center')
    ax.get_legend().remove()

    plt.show()
    print(df2)

def merged_graph(csv,opertaion,graph_title,num):
    nb_mean_row, nb_std_row = get_statistics_results('nb', csv)
    knn_mean_row, knn_std_row = get_statistics_results('knn',csv)
    rf_mean_row, rf_std_row = get_statistics_results('rf', csv)
    ann_mean_row, ann_std_row = get_statistics_results('ann', csv)
    svm_mean_row, svm_std_row = get_statistics_results('svm', csv)
    if opertaion=='mean':
        nb_row=nb_mean_row
        knn_row=knn_mean_row
        rf_row=rf_mean_row
        ann_row=ann_mean_row
        svm_row=svm_mean_row
    else:
        nb_row=nb_std_row
        knn_row=knn_std_row
        rf_row=rf_std_row
        ann_row=ann_std_row
        svm_row=svm_std_row


    try_dict = {
        'method': ['model', 'accuracy', 'tpr', 'fpr', 'precision', 'auc_roc_curve', 'auc_precision_recall',
                   'training_time', 'inference_time'],
        'nb': nb_row,
        'knn': knn_row,
        'rf': rf_row,
        'ann':ann_row,
        'svm': svm_row
    }
    df2 = pd.DataFrame(try_dict,
                       index=['model', 'accuracy', 'tpr', 'fpr', 'precision', 'auc_roc_curve', 'auc_precision_recall',
'training_time', 'inference_time'])

    df2 = df2[1:]
    if num==1:
        ax = df2[:-2].plot(x="method",
                       y=['nb', 'knn', 'rf', 'ann', 'svm'], kind="bar")
    else:
        ax= df2[len(df2)-2:len(df2)].plot(x="method",
                y=['nb', 'knn', 'rf', 'ann', 'svm'], kind="bar")
    plt.xticks(rotation=360)
    plt.title(graph_title,loc='center')

    plt.show()
    print(df2)
# merged_graph('datasets/breast-cancer.csv')
# merged_graph('datasets/bank.csv')
# merged_graph('datasets/acute-nephritis.csv')
# merged_graph('datasets/acute-inflammation.csv')
# merged_graph('datasets/blood.csv')
# merged_graph('datasets/breast-cancer-wisc.csv')
# merged_graph('datasets/breast-cancer-wisc-diag.csv')
# merged_graph('datasets/congressional-voting.csv')
# merged_graph('datasets/breast-cancer-wisc-prog.csv')

curr_csv='datasets/breast-cancer.csv'

#-------------------------------------Single Graphs-------------------------------------

#-------------------------Naive Bayes-------------------------
# single_graph('nb',curr_csv,'mean','Naive Bayes',1)
# single_graph('nb',curr_csv,'mean','Naive Bayes',2)

# single_graph('nb',curr_csv,'std','Naive Bayes',1)
# single_graph('nb',curr_csv,'std','Naive Bayes',2)


#------------------------K Nearest Neighbors-------------------
#single_graph('knn',curr_csv,'mean','K Nearest Neighbors',1)
# single_graph('knn',curr_csv,'mean','K Nearest Neighbors',2)
#
# single_graph('knn',curr_csv,'std','K Nearest Neighbors',1)
# single_graph('knn',curr_csv,'std','K Nearest Neighbors',2)

#-------------------------Random Forest-------------------------
single_graph('rf',curr_csv,'mean','Random Forest',1) #rf-1 Mean
single_graph('rf',curr_csv,'mean','Random Forest',2)

single_graph('rf',curr_csv,'std','Random Forest',1)
single_graph('rf',curr_csv,'std','Random Forest',2)


#-------------------------Artificial Neural Networks-------------------------
# single_graph('ann',curr_csv,'mean','Artificial Neural Networks',1)
# single_graph('ann',curr_csv,'mean','Artificial Neural Networks',2)
#
# single_graph('ann',curr_csv,'std','Artificial Neural Networks',1)
# single_graph('ann',curr_csv,'std','Artificial Neural Networks',2)


#-------------------------Support Vector Machine-------------------------
# single_graph('svm',curr_csv,'mean','Support Vector Machine',1)
# single_graph('svm',curr_csv,'mean','Support Vector Machine',2)
#
# single_graph('svm',curr_csv,'std','Support Vector Machine',1)
# single_graph('svm',curr_csv,'std','Support Vector Machine',2)



#-------------------------------------Merged Graphs-------------------------------------
# merged_graph(curr_csv,'mean',"comparison Between Models",1)
# merged_graph(curr_csv,'mean',"comparison Between Models",2)
#
# merged_graph(curr_csv,'std',"comparison Between Models",1)
# merged_graph(curr_csv,'std',"comparison Between Models",2)

