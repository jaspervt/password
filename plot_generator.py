#Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, roc_auc_score, auc, roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix

def generatePlots(y_test, y_pred):

    # Create confusion matrix
    print("Create confusion matrix")
    df = pd.DataFrame(y_pred, columns = [0,1,2])
    maxValueIndex = df.idxmax(axis=1)
    df2 = pd.DataFrame(y_test, columns = [0,1,2])
    maxValueIndex2 = df2.idxmax(axis=1)
    print("accuracy:")
    print(accuracy_score(maxValueIndex2, maxValueIndex))


    maxValueIndex.to_numpy()
    maxValueIndex2.to_numpy()
    cm = confusion_matrix(maxValueIndex2, maxValueIndex)
    n_classes = 3

    # np.set_printoptions(suppress=True, precision=4)
    f,ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(cm, annot=True, 
                linewidths=0.01,
                linecolor="white", 
                fmt= 'g',ax=ax,cmap="Blues")
    sns.color_palette("rocket", as_cmap=True)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    
    # Calculate error measurements
    print("Calculate error measurements")
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
        #prc = precision_recall_curve(maxValueIndex2[0][:], maxValueIndex[0][:])
        #aps = average_precision_score 
        #ras = roc_auc_score 
        #auc = auc(specificity, recall) 

        print("for class {}: accuracy {}, recall {}, specificity {}\
            precision {}, f1 {}".format(c,round(accuracy,4),round(recall,4), round(specificity,4), round(precision,4),round(f1_score,4)))



    # Creat ROC Strength plots
    print("Create ROC Strength plots")
    for i in range(0,3):
        y_test2 = y_test[:,i]
        y_pred2 = y_pred[:,i]

        false_positive_rate, recall, thresholds = roc_curve(y_test2,y_pred2)
        roc_auc = auc(false_positive_rate, recall)
        plt.figure()
        plt.title('ROC Strength {}'.format(i))
        plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1], [0,1], 'r--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.ylabel('Recall')
        plt.xlabel('Fall-out (1-Specificity)')
        
    plt.show()


def generate_per_epoch(y_test, y_pred,train_hist,epochs):
    plt.figure(figsize=(10,4))
    plt.plot(train_hist[['loss','val_loss']])
    plt.legend(['loss','val_loss'])
    plt.title('Loss Per Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    #plot accuracy per epoch
    plt.figure(figsize=(10,4))
    plt.plot(train_hist[['accuracy','val_accuracy']])
    plt.legend(['accuracy','val_accuracy'])
    plt.title('Accuracy Per Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

    generatePlots(y_test, y_pred)