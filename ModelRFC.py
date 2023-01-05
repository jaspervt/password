#Libraries
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, roc_auc_score, auc, roc_curve

data = pd.read_csv("output.csv")
data.dropna()

# y = strength(nieuw) |||| x = length, numdigits,num_upper en numlower
# wat voor , is voor rows selecteren na , is voor kolommen
# Alle features
x = data.iloc[:, [3,4,6,7,8,9,10,11,12,13,16,17,20,21,22,23,25]]
# print(x)
# print(x.shape)
# test = data.iloc[1:6,[2,3,5,6,7,8,9,10,11,12]]
# print(test)
length_x = x.shape[1]
#1:2 = oude strekte, 11:12 = nieuwe sterkte
y = data.iloc[:,5:6]
# print(y)

#Normalizing/scaling the data
sc = StandardScaler()
x = sc.fit_transform(x)
# print(x.shape)
# print(x)

#hierdoor krijg je array speciaal voor bepaalde rating
#strength 3 = [0. 0. 0. 1. 0.] en strength 1 = [0. 1. 0. 0. 0.] en strength 0 = [1. 0. 0. 0. 0.]
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()
# print(y)

#split train en test data train_data = 90% van de data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.05)

model = RandomForestClassifier()
model.fit(x_train,y_train)
print(model.score(x_test, y_test))
y_pred = model.predict(x_test)
y_test0 = y_test[:,0]
y_pred0 = y_pred[:,0]
y_test1 = y_test[:,1]
y_pred1 = y_pred[:,1]
y_test2 = y_test[:,2]
y_pred2 = y_pred[:,2]
# print(y_pred)
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# print(model.predict(test))
df = pd.DataFrame(y_pred, columns = [0,1,2])
maxValueIndex = df.idxmax(axis=1)
df2 = pd.DataFrame(y_test, columns = [0,1,2])
maxValueIndex2 = df2.idxmax(axis=1)
#print(maxValueIndex)
#,validation_data=(x_test,y_test)
maxValueIndex.to_numpy()
maxValueIndex2.to_numpy()
print(maxValueIndex2)
cm = confusion_matrix(maxValueIndex2, maxValueIndex)
print(cm)
n_classes = 3
y_pred = np.argmax(y_pred, axis=1)
f,ax = plt.subplots(figsize=(12, 12))
sns.heatmap(cm, annot=True, 
            linewidths=0.01,
            linecolor="white", 
            fmt= '.1f',ax=ax,cmap="Blues")
sns.color_palette("rocket", as_cmap=True)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
# np.set_printoptions(suppress=True, precision=4)


for c in range(n_classes):
    tp = cm[c,c]
    fp = sum(cm[:,c]) - cm[c,c]
    fn = sum(cm[c,:]) - cm[c,c]
    tn = sum(np.delete(sum(cm)-cm[c,:],c))

    accuracy = (tp+tn)/(tp+tn+fn+fp)
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    specificity = tn/(tn+fp)
    f1_score = 2*((precision*recall)/(precision+recall))
    

    #print(f"for class {c}: acc {accuracy}, recall {recall},\
    #      precision {precision}, f1 {f1_score}")
    print("for class {}: accuracy {}, recall {}, specificity {}\
          precision {}, f1 {}".format(c,round(accuracy,4),round(recall,4), round(specificity,4), round(precision,4),round(f1_score,4)))

false_positive_rate, recall, thresholds = roc_curve(y_test0,y_pred0)
roc_auc = auc(false_positive_rate, recall)
plt.figure()
plt.title('ROC Strength 0')
plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out (1-Specificity)')

false_positive_rate, recall, thresholds = roc_curve(y_test1,y_pred1)
roc_auc = auc(false_positive_rate, recall)
print(roc_auc)
print(false_positive_rate, recall, thresholds)
plt.figure()
plt.title('ROC Strength 1')
plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out (1-Specificity)')

false_positive_rate, recall, thresholds = roc_curve(y_test2,y_pred2)
roc_auc = auc(false_positive_rate, recall)
print(roc_auc)
print(false_positive_rate, recall, thresholds)
plt.figure()
plt.title('ROC Strength 2')
plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out (1-Specificity)')
plt.show()   
    
#0.7298