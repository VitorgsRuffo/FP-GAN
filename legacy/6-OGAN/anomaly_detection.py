from import_data import import_gan_testing_data
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
import matplotlib.pyplot as plt


# step 0: import data
anomalous_day_2, labels = import_gan_testing_data('orion')



# step 1: load trained discriminators
bits_discriminator             = load_model('./model/bits_discriminator.h5', compile=False)
dst_ip_entropy_discriminator   = load_model('./model/dst_ip_entropy_discriminator.h5', compile=False)
dst_port_entropy_discriminator = load_model('./model/dst_port_entropy_discriminator.h5', compile=False)

src_ip_entropy_discriminator   = load_model('./model/src_ip_entropy_discriminator.h5', compile=False)
src_port_entropy_discriminator = load_model('./model/src_port_entropy_discriminator.h5', compile=False)
packets_discriminator          = load_model('./model/packets_discriminator.h5', compile=False)


# step 2: make predictions on unseen traffic data
# predictions[predictions < 0.5] = 1
# predictions[predictions >= 0.5] = 0




bits_predictions = bits_discriminator.predict(
    np.concatenate((anomalous_day_2[:, 0:1], anomalous_day_2[:, 1:2]), axis=1))

predictions = np.zeros((bits_predictions.shape))

predictions[bits_predictions <= 0.5] = predictions[bits_predictions <= 0.5] + 1



dst_ip_entropy_predictions = dst_ip_entropy_discriminator.predict(
    np.concatenate((anomalous_day_2[:, 0:1], anomalous_day_2[:, 2:3]), axis=1))

predictions[dst_ip_entropy_predictions <= 0.5] = predictions[dst_ip_entropy_predictions <= 0.5] + 1



dst_port_entropy_predictions = dst_port_entropy_discriminator.predict(
    np.concatenate((anomalous_day_2[:, 0:1], anomalous_day_2[:, 3:4]), axis=1))

predictions[dst_port_entropy_predictions <= 0.5] = predictions[dst_port_entropy_predictions <= 0.5] + 1



src_ip_entropy_predictions = src_ip_entropy_discriminator.predict(
    np.concatenate((anomalous_day_2[:, 0:1], anomalous_day_2[:, 4:5]), axis=1))

predictions[src_ip_entropy_predictions <= 0.5] = predictions[src_ip_entropy_predictions <= 0.5] + 1



src_port_entropy_predictions = src_port_entropy_discriminator.predict(
    np.concatenate((anomalous_day_2[:, 0:1], anomalous_day_2[:, 5:6]), axis=1))

predictions[src_port_entropy_predictions <= 0.5] = predictions[src_port_entropy_predictions <= 0.5] + 1



packets_predictions = packets_discriminator.predict(
    np.concatenate((anomalous_day_2[:, 0:1], anomalous_day_2[:, 6:7]), axis=1))

predictions[packets_predictions <= 0.5] = predictions[packets_predictions <= 0.5] + 1


predictions[predictions < 6] = 0
predictions[predictions == 6] = 1 #6 discriminators predicted the second as anomoulous, so we consider it anomalous...





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


# Performance metrics - ano_day_1 (150 epoch, 1-feature threshold: 0.5, 6-feature threshold: 6): 


# Confusion matrix: 
# [[ 4480   714]
#  [ 4222 76984]]


# Precision: 0.862533692722372

# Recall: 0.5148241783498047

# F1-score: 0.6447898675877951


# Performance metrics - ano_day_2 (150 epoch, 1-feature threshold: 0.5, 6-feature threshold: 6):  


# Confusion matrix: 
# [[  607  1194]
#  [ 9175 75424]]


# Precision: 0.337034980566352

# Recall: 0.06205274994888571

# F1-score: 0.10480877147543814
