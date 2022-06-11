import statistics
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

from naive_bayes import get_gnb_results

sns.set()




def get_statistics_results(model,csv):
    if model == 'nb':
        accuracy_lst, tpr_lst, fpr_lst, precision_lst, auc_roc_curve_lst, auc_precision_recall_lst, training_time_lst, inference_time_lst = get_gnb_results(csv)
    elif model == 'knn':
        pass
    elif model=='rf':
        pass
    elif model=='ann':
        pass
    elif model=='svm':
        pass
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
    knn_mean_row,knn_std_row=get_statistics_results('nb',csv)
    rf_mean_row,rf_std_row=get_statistics_results('nb',csv)
    ann_mean_row,ann_std_row=get_statistics_results('nb',csv)
    svm_mean_row,svm_std_row=get_statistics_results('nb',csv)

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
nb_mean_row, nb_std_row = get_statistics_results('nb','all_datasets/breast-cancer.csv')
knn_mean_row,knn_std_row=get_statistics_results('nb','all_datasets/breast-cancer.csv')
rf_mean_row,rf_std_row=get_statistics_results('nb','all_datasets/breast-cancer.csv')
ann_mean_row,ann_std_row=get_statistics_results('nb','all_datasets/breast-cancer.csv')
svm_mean_row,svm_std_row=get_statistics_results('nb','all_datasets/breast-cancer.csv')
try_dict={
    'method':['model','accuracy','tpr','fpr','precision','auc_roc_curve','auc_precision_recall','training_time','inference_time'],
    'nb':nb_mean_row,
    'knn':knn_mean_row,
    'rf':rf_mean_row,
    'ann':ann_mean_row,
    'svm':svm_mean_row
}
df2=pd.DataFrame(try_dict,index=['model','accuracy','tpr','fpr','precision','auc_roc_curve','auc_precision_recall','training_time','inference_time'])


df2=df2[1:]
ax = df2[:-2].plot(x="method",
             y=['nb', 'knn', 'rf', 'ann', 'svm'], kind="bar")
#ax= df2[len(df2)-2:len(df2)].plot(x="method",
#             y=['nb', 'knn', 'rf', 'ann', 'svm'], kind="bar")
plt.xticks(rotation=360)

plt.show()
print(df2)