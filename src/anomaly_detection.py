from import_data import import_orion_anomalous_data, import_cic_anomalous_data
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
from pickle import load


# step 0: import data
#anomalous_day, labels = import_orion_anomalous_data(2, False)
anomalous_day, labels = import_cic_anomalous_data(2)


# step 1: load trained discriminator
discriminator = load_model('./model/discriminator.h5', compile=False)


# step 2: load found threshold
_file = open('threshold.pkl', 'rb')
threshold = load(_file)


# step 3: make predictions on unseen traffic data
predictions = discriminator.predict(anomalous_day)



plt.rcParams['figure.figsize'] = [10.80,7.20]
plt.plot([i for i in range(0, predictions.shape[0])], predictions)
plt.xlabel('second')
plt.ylabel('prediction')
plt.show()
# plt.savefig(f"./disc_pred.png", dpi=800)
# plt.close()


# step 4: discriminate normal from anomalous data based on previously calculated threshold.
normal_mean_array = [threshold['nmean']] * predictions.shape[0]
normal_mean_array = np.array(normal_mean_array)
normal_mean_array = np.reshape(normal_mean_array, (-1, 1))
distance = abs(predictions - normal_mean_array)
predictions = np.zeros_like(distance)
predictions[distance > threshold['th']] = 1



# step 3: calculate performance metrics
def calculate_metrics(predictions, labels):
    print(f"\n\nPerformance metrics - anomalous day 2: ")

    figure, axis = plt.subplots(1, 2, constrained_layout=True)
    #plotando o grafico dos rotulos do conjunto de treinamento
    axis[0].set_title('Actual labels')
    axis[0].step(np.linspace(0, labels.shape[0], labels.shape[0]), labels, color='green')
    #plotando o grafico das previsoes
    axis[1].set_title('Predicted labels')
    axis[1].step(np.linspace(0, predictions.shape[0], predictions.shape[0]), predictions, color='red')
    #plt.axvspan(10000, 20000, color='r', alpha=0.5)
    plt.savefig(f"./predictions.png", dpi=800)
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


calculate_metrics(predictions, labels)
