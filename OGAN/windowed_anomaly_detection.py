from import_data import import_gan_testing_data
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


# step 0: import data
window_size = 5
anomalous_day_1_x, anomalous_day_1_y, labels_1, anomalous_day_2_x, anomalous_day_2_y, labels_2 = import_gan_testing_data('orion', 'gru', window_size)


# step 1: load trained discriminator
discriminator = load_model('./model/discriminator.h5', compile=False)

# step 2: make predictions on unseen traffic data
pred_1 = discriminator.predict(anomalous_day_1_x)
pred_2 = discriminator.predict(anomalous_day_2_x)

# print(pred_1.shape)

# unique, counts = np.unique(pred_1, return_counts=True)
# print(dict(zip(unique, counts)))

# unique, counts = np.unique(labels_1, return_counts=True)
# print(dict(zip(unique, counts)))

#input()



predictions_1 = np.zeros_like(labels_1)
for i in range(len(pred_1)):
    window_start = i
    window_end = window_start+window_size
    #print(f"\n\ns: {window_start}; e: {window_end}\n")
    #print(pred_1[i, 0])
    if pred_1[i, 0] >= 0.5:
        predictions_1[window_start:window_end] = 1



# unique, counts = np.unique(predictions_1, return_counts=True)
# print(dict(zip(unique, counts)))

# input()

predictions_2 = np.zeros_like(labels_2)
for i in range(len(pred_2)):
    window_start = i
    window_end = window_start+window_size
    if pred_2[i, 0] >= 0.5:
        predictions_2[window_start:window_end] = 1



# predictions_1[predictions_1 >= 0.5] = 1 
# predictions_1[predictions_1 < 0.5] = 0

# predictions_2[predictions_2 >= 0.5] = 1
# predictions_2[predictions_2 < 0.5] = 0







# step 3: calculate performance metrics
def calculate_metrics(predictions, labels, day):
    print(f"\n\nPerformance metrics - anomalous day {day}: ")

    figure, axis = plt.subplots(1, 2, constrained_layout=True)
    #plotando o grafico dos rotulos do conjunto de treinamento
    axis[0].set_title('Actual labels')
    axis[0].step(np.linspace(0, 86400, 86400), labels, color='green')
    #plotando o grafico das previsoes
    axis[1].set_title('Predicted labels')
    axis[1].step(np.linspace(0, 86400, 86400), predictions, color='red')
    #plt.axvspan(10000, 20000, color='r', alpha=0.5)
    plt.show()

    print("\n\nConfusion matrix: ")
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    print(np.array([[tp, fp], [fn, tn]]))

    print(f"\n\nPrecision: {precision_score(labels, predictions)}")
    print(f"\nRecall: {recall_score(labels, predictions)}")
    print(f"\nF1-score: {f1_score(labels, predictions)}")
    print(f"\nMatthews Correlation Coefficient (MCC): {matthews_corrcoef(labels, predictions)}")

    #ROC curve
    fpr, tpr, _ = metrics.roc_curve(labels,  predictions)
    auc = metrics.roc_auc_score(labels, predictions)
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()


calculate_metrics(predictions_1, labels_1, 1)
calculate_metrics(predictions_2, labels_2, 2)
