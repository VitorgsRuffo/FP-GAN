from import_data import import_gan_testing_data
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt


# step 0: import data
anomalous_day_1, labels_1, anomalous_day_2, labels_2 = import_gan_testing_data('orion')


# step 1: load trained discriminator
discriminator = load_model('./model/discriminator.h5', compile=False)


# step 2: make predictions on unseen traffic data
predictions_1 = discriminator.predict(anomalous_day_1)
predictions_2 = discriminator.predict(anomalous_day_2)

scaler_1 = MinMaxScaler((0, 1)).fit(predictions_1)
predictions_1 = scaler_1.transform(predictions_1)

scaler_2 = MinMaxScaler((0, 1)).fit(predictions_2)
predictions_2 = scaler_2.transform(predictions_2)

plt.rcParams['figure.figsize'] = [10.80,7.20]

plt.plot([i for i in range(0, predictions_1.shape[0])], predictions_1)
plt.xlabel('second')
plt.ylabel('prediction')
plt.show()
# plt.savefig(f"./pred_1.png", dpi=800)
# plt.close()

plt.plot([i for i in range(0, predictions_2.shape[0])], predictions_2)
plt.xlabel('second')
plt.ylabel('prediction')
plt.show()
# plt.savefig(f"./pred_2.png", dpi=800)
# plt.close()



 
input()


predictions_1[predictions_1 >= 0.52] = 1 #is this threshold correct.
predictions_1[predictions_1 < 0.52] = 0

predictions_2[predictions_2 >= 0.52] = 1
predictions_2[predictions_2 < 0.52] = 0






# step 3: calculate performance metrics
def calculate_metrics(predictions, labels, day):
    print(f"\n\nPerformance metrics - anomalous day {day}: ")

    figure, axis = plt.subplots(1, 2, constrained_layout=True)
    #plotando o grafico dos rotulos do conjunto de treinamento
    axis[0].set_title('Actual labels')
    axis[0].step(np.linspace(0, labels.shape[0], labels.shape[0]), labels, color='green')
    #plotando o grafico das previsoes
    axis[1].set_title('Predicted labels')
    axis[1].step(np.linspace(0, predictions.shape[0], predictions.shape[0]), predictions, color='red')
    #plt.axvspan(10000, 20000, color='r', alpha=0.5)
    plt.savefig(f"./predictions-{day}.png", dpi=800)
    plt.close()

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
