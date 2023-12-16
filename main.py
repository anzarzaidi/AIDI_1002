import random_forest_5fold_param_tuning
import train_bnb
import train_cnb
import train_decision_tree
import train_dnn
import train_gnb
import train_gnb_5fold
import train_knn
import train_knn_5fold
import train_lr
import train_mnb
import train_randomforest
import train_svm
import train_svm_10fold
import train_svm_5fold

print("Executing existing code")
print("###############################################################")
train_svm.execute(False)
train_svm_5fold.execute()
train_gnb.execute(False)
train_gnb_5fold.execute()
train_knn.execute(False)
train_knn_5fold.execute()
print("###############################################################")


print("Executing New code")

train_bnb.execute(False)
train_cnb.execute()
train_decision_tree.execute()
train_dnn.execute()
train_lr.execute()
train_mnb.execute()
train_randomforest.execute()
random_forest_5fold_param_tuning.execute()
train_svm_10fold.execute()
print("###############################################################")