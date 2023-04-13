### Calculate best threshold for discriminating normal traffic from anomalous one.
from import_data import import_orion_anomalous_data, import_orion_normal_data
from import_data import import_cic_anomalous_data, import_cic_normal_data
import numpy as np
from tensorflow.keras.models import load_model
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


plt.rcParams['figure.figsize'] = [10.80,7.20]
mpl.rcParams['lines.linewidth'] = 0.6


# Step 0: import traffic data and trained discriminator
import sys
if len(sys.argv) <= 1:
    print("Please.\nSpecify the dataset:\n\t(orion1 | orion2)\n\tcic\n")
    quit(1)
dataset = sys.argv[1]

if dataset == 'orion1':
    normal_day, scaler = import_orion_normal_data(dataset=1)
    anomalous_day, labels = import_orion_anomalous_data(dataset=1, day=1, portscan=False)

elif dataset == 'orion2':
    normal_day, scaler = import_orion_normal_data(dataset=2)
    anomalous_day, labels = import_orion_anomalous_data(dataset=2, day=1, portscan=False)

else:
    normal_day, scaler = import_cic_normal_data()
    anomalous_day, labels = import_cic_anomalous_data(day=1)

    
discriminator = load_model('./model/discriminator')



# Step 1: calculate normal data predictions mean
normal_predictions = discriminator.predict(normal_day)
normal_mean = np.mean(normal_predictions, dtype = np.float64, axis=0)
normal_mean = normal_mean[0]
#std_dev = np.std(normal_predictions, dtype = np.float64, axis=0)

x = [i for i in range(0, normal_predictions.shape[0])]
plt.plot(x, normal_predictions, color='#205295')

y1 = [normal_mean] * normal_predictions.shape[0]
mpl.rcParams['lines.linewidth'] = 1.5
plt.plot(x, y1, '-g', label=f"média: {round(normal_mean, 4)}")

plt.xlabel('segundo')
plt.ylabel('saída do discriminador')
#plt.yticks(np.arange(round(normal_mean, 2)-0.05, 0.55, step=0.02))
plt.yticks(np.arange(0.48, 0.51, step=0.01))
plt.ylim((0.48, 0.51))
plt.legend(loc='upper right')
plt.margins(x=0)
plt.savefig(f"./th_disc_normal_pred.png", dpi=800)
plt.close()



# Step 2: predict anomalous danomalous_day, labels = import_orion_anomalous_datata discriminating score
anomalous_predictions = discriminator.predict(anomalous_day)

# mpl.rcParams['lines.linewidth'] = 0.6
# plt.plot([i for i in range(0, anomalous_predictions.shape[0])], anomalous_predictions, color='#205295', label=f"Mean={normal_mean}")
# plt.xlabel('second')
# plt.ylabel('prediction')
# plt.legend(loc=1)
# plt.margins(x=0)
# plt.savefig(f"./th_disc_anomalous_pred.png", dpi=800)
# plt.close()


# Step 3: calculate distance between the scores and normal data mean
normal_mean_array = [normal_mean] * anomalous_predictions.shape[0]
normal_mean_array = np.array(normal_mean_array)
normal_mean_array = np.reshape(normal_mean_array, (-1, 1))
#distances = abs(anomalous_predictions - normal_mean_array)
distances = anomalous_predictions - normal_mean_array
distances[distances<0] = 0 #everything below mean is considered normal

    

# Step 4: search for the best threshold (the one which maximizes the model's MCC)
# Finds a threshold the limits the distance that a data point prediction must have from the normal data prediction mean 
# in order to be considered normal. Data points which predictions distances lie outside that threshold are said to be anomalous.
best_threshold = -1
best_mcc = -1

interval = np.linspace(0.0, 0.1, num=100)
for threshold in interval:
    
    predictions = np.zeros_like(distances)

    predictions[distances > threshold] = 1

    mcc = metrics.matthews_corrcoef(labels, predictions)

    if mcc > best_mcc:
        best_mcc = mcc
        best_threshold = threshold
print(f'\n\nFound threshold: {best_threshold}\nmcc:{best_mcc}')



# Step 5: save found threshold
threshold = {
    'th': best_threshold,
    'nmean': normal_mean
}

from pickle import dump
_file = open('threshold.pkl', 'wb')
dump(threshold, _file)
_file.close()
