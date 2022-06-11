from knn import print_knn_results
from naive_bayes import print_gnb_results

print("--------------------------Naive Bayes--------------------------")
print_gnb_results('datasets/breast-cancer.csv')
print_gnb_results('datasets/bank.csv')
print_gnb_results('datasets/acute-nephritis.csv')
print_gnb_results('datasets/acute-inflammation.csv')
print_gnb_results('datasets/blood.csv')
print_gnb_results('datasets/breast-cancer-wisc.csv')
print_gnb_results('datasets/breast-cancer-wisc-diag.csv')
print_gnb_results('datasets/congressional-voting.csv')
print_gnb_results('datasets/chess-krvkp.csv')
print_gnb_results('datasets/breast-cancer-wisc-prog.csv')


print_gnb_results()
print("-------------------------- KNN --------------------------")
print_knn_results()