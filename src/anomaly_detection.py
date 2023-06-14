from import_data import import_orion_anomalous_data, import_cic_anomalous_data, import_orion_normal_data
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
import matplotlib as mpl
from pickle import dump, load

plt.rcParams['figure.figsize'] = [10.80,7.20]

# step 0: import data
import sys
if len(sys.argv) <= 1:
    print("Please.\nSpecify the dataset:\n\t(orion1 | orion2)\n\tcic\n")
    quit(1)
dataset = sys.argv[1]

anomalous_day, labels = None, None
if dataset == 'orion1':
    normal_day, scaler = import_orion_normal_data(dataset=1)
    anomalous_day, labels = import_orion_anomalous_data(dataset=1, day=2, portscan=False)

elif dataset == 'orion2':
    normal_day, scaler = import_orion_normal_data(dataset=2)
    anomalous_day, labels = import_orion_anomalous_data(dataset=2, day=2, portscan=False)

else:
    anomalous_day, labels = import_cic_anomalous_data(day=2)



# step 1: load trained discriminator
discriminator = load_model('./model/discriminator')


# step 2: load found threshold
_file = open('threshold.pkl', 'rb')
threshold = load(_file)




# step 3: make predictions on unseen traffic data
predictions = discriminator.predict(anomalous_day)


#plotting predictions
import locale
plt.rcParams['axes.formatter.use_locale'] = True


x = [i for i in range(0, predictions.shape[0])]

mpl.rcParams['lines.linewidth'] = 0.6
plt.plot(x, predictions, color='darkgreen')

y1 = [threshold['nmean']] * predictions.shape[0]
mpl.rcParams['lines.linewidth'] = 1.5
plt.plot(x, y1, color='orange', linestyle='solid', label=f"média: {round(threshold['nmean'], 4)}".replace(".", ","))


y2 = [threshold['nmean'] + threshold['th']] * predictions.shape[0]
plt.plot(x, y2, color="orange", linestyle="dashed", label=f"limiar: {round(threshold['th'], 4)}".replace(".", ","))


plt.fill_between(
        x, 
        y1,
        y2, 
        color= "orange",
        alpha= 0.2)


li = list(labels)
start = li.index(1)
end = start + (li.count(1)-1) 
#plt.plot(x[start:end], predictions[start:end], color='darkred')
#plt.fill_between(
#    x[start:end], 
#    list(predictions[start:end].reshape(-1)),
#    0.40,
#    color= "r",
#    alpha= 0.4,
#    label="Intervalo anômalo")

plt.axvspan(start, end, label="Intervalo anômalo", color='r', alpha=0.4)



locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
import matplotlib.ticker as tkr
def func(x, pos):  # formatter function takes tick label and tick position
    return locale.format_string("%.2f", x)
axis_format = tkr.FuncFormatter(func)  # make formatter

plt.xlabel('segundo')
plt.ylabel('saída do descriminador')
#plt.yticks(np.arange(round(threshold['nmean'], 2)-0.05, 0.65, step=0.02))
plt.ylim((0.40, 0.57))
ax = plt.gca()
ax.yaxis.set_major_formatter(axis_format) #when using ','

plt.legend(loc='upper left')
plt.margins(x=0)
plt.savefig(f"./disc_pred.png", dpi=800)
plt.close()




# step 4: discriminate normal from anomalous data based on previously calculated threshold.
normal_mean_array = [threshold['nmean']] * predictions.shape[0]
normal_mean_array = np.array(normal_mean_array)
normal_mean_array = np.reshape(normal_mean_array, (-1, 1))
#distances = abs(predictions - normal_mean_array)

distances = predictions - normal_mean_array
distances[distances<0] = 0 #everything below mean is considered benign





predictions = np.zeros_like(distances)
predictions[distances > threshold['th']] = 1



# step 3: calculate performance metrics
def calculate_metrics(raw_predictions, predictions, labels):
    print(f"\n\nPerformance metrics - anomalous day 2: ")

    plt.rcParams['figure.figsize'] = [10.80,7.20]
    figure, axis = plt.subplots(2, 1, constrained_layout=True)
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
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib as mpl
    scaler = MinMaxScaler((0, 1)).fit(raw_predictions.reshape(-1, 1)) 
    normalized_raw_predictions = scaler.transform(raw_predictions.reshape(-1, 1))
    mpl.rcParams['lines.linewidth'] = 1

    _file = open('gan_normalized_raw_predictions.pkl', 'wb')
    dump(normalized_raw_predictions, _file)
    _file.close()

    fpr, tpr, _ = metrics.roc_curve(labels,  normalized_raw_predictions)
    auc = metrics.roc_auc_score(labels, normalized_raw_predictions)
    plt.plot(fpr,tpr,label="NIDS-GAN\n(AUROC="+str(round(auc, 4)).replace('.', ',')+")", color="#0066CC")
    x = np.linspace(0, 1, 50)
    plt.plot(x,x,linestyle='dashed', label="Classificador aleatório\n(AUROC=0,5000)", color="black")

    plt.ylabel('True Positive Rate (Recall)')
    plt.xlabel('False Positive Rate (FPR)')
    plt.legend(loc=4)
    plt.savefig("./roc_gan.jpg", format='jpg', dpi=800)
    plt.close()



calculate_metrics(distances, predictions, labels)



# Step 4.1: scale data based on distance from mean and hyperparameter k
# k = 20
# scaling_array = np.ones_like(predictions)
# distances = distances * k
# scaling_array = scaling_array + distances
# scaled_predictions = predictions * scaling_array
# distances = abs(scaled_predictions - normal_mean_array)


# plt.rcParams['figure.figsize'] = [10.80,7.20]
# plt.plot([i for i in range(0, scaled_predictions.shape[0])], scaled_predictions, label=f"Mean={threshold['nmean']}\nth={threshold['th']}")
# plt.xlabel('second')
# plt.ylabel('prediction')
# plt.legend(loc=1)
# plt.savefig(f"./disc_scaled_pred.png", dpi=800)
# plt.close()
