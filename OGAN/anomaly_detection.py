from import_data import import_gan_testing_data
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
import matplotlib.pyplot as plt


# step 0: import data
anomalous_day_2, labels, scaler = import_gan_testing_data('orion')


# step 1: load trained discriminator
discriminator = load_model('./model/discriminator.h5', compile=False)


# step 2: make predictions on unseen traffic data
predictions = discriminator.predict(anomalous_day_2)
predictions[predictions >= 0.5] = 1
predictions[predictions < 0.5] = 0


# step 3: calculate performance metrics
print("\n\nPerformance metrics: ")

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
