
import train_bnb
import train_gnb
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
import train_rf_single
import train_svm
import train_svm_5fold
import train_svm_10fold


print("Executing existing code")
print("###############################################################")
train_svm.execute()
train_svm_5fold.execute()
train_gnb.execute()
train_gnb_5fold.execute()
train_knn.execute()
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
train_rf_single.execute()
train_svm_10fold.execute()