from naive_bayes import print_gnb_results
from knn import print_knn_results
from random_forest import print_rf_results
from svm import print_svm_results

print("--------------------------Naive Bayes--------------------------")
print_gnb_results('all_datasets/breast-cancer.csv')
# print_gnb_results('all_datasets/bank.csv')
# print_gnb_results('all_datasets/acute-nephritis.csv')
# print_gnb_results('all_datasets/acute-inflammation.csv')
# print_gnb_results('all_datasets/blood.csv')
# print_gnb_results('all_datasets/breast-cancer-wisc.csv')
# print_gnb_results('all_datasets/breast-cancer-wisc-diag.csv')
# print_gnb_results('all_datasets/congressional-voting.csv')
# print_gnb_results('all_datasets/chess-krvkp.csv')
# print_gnb_results('all_datasets/breast-cancer-wisc-prog.csv')


print("--------------------------Random Forest--------------------------")
print_rf_results('datasets/breast-cancer.csv')
# print_rf_results('datasets/bank.csv')
# print_rf_results('datasets/acute-nephritis.csv')
# print_rf_results('datasets/acute-inflammation.csv')
# print_rf_results('datasets/blood.csv')
# print_rf_results('datasets/breast-cancer-wisc.csv')
# print_rf_results('datasets/breast-cancer-wisc-diag.csv')
# print_rf_results('datasets/congressional-voting.csv')
# print_rf_results('datasets/chess-krvkp.csv')
# print_rf_results('datasets/breast-cancer-wisc-prog.csv')

print("--------------------------SVM--------------------------")
print_svm_results()


print("-------------------------- KNN --------------------------")
print_knn_results()
