# Módulo de detecção.
#
# Baseando-se no comportamento normal aprendido atraves de dados historicos da rede, 
# o sistema pode começar a fazer previsões sobre o comportamento esperado da rede em diferentes 
# intervalos de um certo dia. Com essas previsões em mãos, ele as compara com o comportamento real 
# da rede. Caso a diferença entre eles sejam muito discrepantes (fora do threshold)
# um alarme é soado para o gerente da rede, indicando uma anomalia.

import numpy as np
from keras.models import load_model
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
from pickle import load
from import_data import import_orion_normal_windowed_data, import_orion_anomalous_windowed_data
from utils import plot_prediction_real_graph


#
## Passo 1, Carregar o modelo treinado e o threshold ideal...
#
model = load_model('gru')

_file = open('threshold.pkl', 'rb')
threshold = load(_file)


#
## Passo 1, Carregar os dados de teste...
#

window_size = 10 
portscan = False


anomalous_data_2_x, anomalous_data_2_y,\
anomalous_data_2_labels = import_orion_anomalous_windowed_data(dataset=1, day=2, portscan=portscan, window_size=window_size)



#
## Passo 2, Testar o modelo em um novo dia COM anomalias usando o threshold encontrado:
#


prediction = model.predict(anomalous_data_2_x) # shape: (86395, 6)


#
# 2.1 Plotando os graficos que mostram a previsão do modelo vs os dados reais. 
# Assim, podemos vizualizar se a caracterização do modelo esta boa. Isto é, se ele esta
# conseguindo realizar previsões proximas da realidade.
#

if portscan:
    plot_prediction_real_graph("Prediction-real graph:\n171218_portscan_ddos", 
                            "./171218_portscan_ddos_prediction_real_graph.jpg",
                            prediction, anomalous_data_2_y, [('09:45:00', '11:10:00'), ('17:37:00', '18:55:00')], window_size)
else:
    plot_prediction_real_graph("Prediction-real graph:\n171218_portscan_ddos", 
                           "./171218_portscan_ddos_prediction_real_graph.jpg",
                           prediction, anomalous_data_2_y, [('17:37:00', '18:55:00')], window_size)


#
# 2.2 Detecção:
#
bytes_count_abs_error   = abs(prediction[:, 0] - anomalous_data_2_y[:, 0])
dst_ip_ent_abs_error    = abs(prediction[:, 1] - anomalous_data_2_y[:, 1])
dst_port_ent_abs_error  = abs(prediction[:, 2] - anomalous_data_2_y[:, 2])
src_ip_ent_abs_error    = abs(prediction[:, 3] - anomalous_data_2_y[:, 3])
src_port_ent_abs_error  = abs(prediction[:, 4] - anomalous_data_2_y[:, 4])
packets_count_abs_error = abs(prediction[:, 5] - anomalous_data_2_y[:, 5])

abs_error = dst_ip_ent_abs_error + dst_port_ent_abs_error + \
            src_ip_ent_abs_error + src_port_ent_abs_error + \
            bytes_count_abs_error + packets_count_abs_error  # shape: (86380, 1)

predicted_labels = np.zeros_like(abs_error)
predicted_labels[abs_error > threshold] = 1


# Performance:
print('\n\nDay 2 Metrics:')

plt.rcParams['figure.figsize'] = [10.80,7.20]
figure, axis = plt.subplots(2, 1, constrained_layout=True)
#plotando o grafico dos rotulos do conjunto de treinamento
axis[0].set_title('Actual labels')
axis[0].step(np.linspace(0, anomalous_data_2_labels.shape[0], anomalous_data_2_labels.shape[0]), anomalous_data_2_labels, color='green')
#plotando o grafico das previsoes
axis[1].set_title('Predicted labels')
axis[1].step(np.linspace(0, predicted_labels.shape[0], predicted_labels.shape[0]), predicted_labels, color='red')
#plt.axvspan(10000, 20000, color='r', alpha=0.5)
plt.savefig(f"./predictions.png", dpi=800)
plt.close()

print("Confusion matrix: ")
tn, fp, fn, tp = confusion_matrix(anomalous_data_2_labels, predicted_labels).ravel()
print(np.array([[tp, fp], [fn, tn]]))

print(f"\n\nPrecision: {precision_score(anomalous_data_2_labels, predicted_labels)}")

print(f"\nRecall: {recall_score(anomalous_data_2_labels, predicted_labels)}")

print(f"\nF1-score: {f1_score(anomalous_data_2_labels, predicted_labels)}")

print(f"\nMatthews Correlation Coefficient (MCC): {matthews_corrcoef(anomalous_data_2_labels, predicted_labels)}")

#ROC curve:
fpr, tpr, _ = metrics.roc_curve(anomalous_data_2_labels,  predicted_labels)
auc = metrics.roc_auc_score(anomalous_data_2_labels, predicted_labels)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()
plt.close()
