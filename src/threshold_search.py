### Calculate best threshold for discriminating normal traffic from anomalous one.
from import_data import import_orion_anomalous_data, import_orion_normal_data
import numpy as np
from tensorflow.keras.models import load_model
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef


# plt.plot([i for i in range(0, predictions.shape[0])], predictions, label=f"Mean={mean}\nstd_dev={std_dev}")
# plt.xlabel('second')
# plt.ylabel('prediction')
# plt.legend(loc=1)
# plt.show()



# Step 0: import traffic data and trained discriminator
normal_day, scaler = import_orion_normal_data()
anomalous_day, labels = import_orion_anomalous_data(1, False)
discriminator = load_model('./model/discriminator.h5', compile=False)


# Step 1: calculate normal data predictions mean
predictions = discriminator.predict(normal_day)
normal_mean = np.mean(predictions, dtype = np.float64, axis=0)


# Step 2: predict anomalous data discriminating score
predictions = discriminator.predict(anomalous_day)


# Step 3: calculate distance between the scores and normal data mean
normal_mean_array = [normal_mean] * predictions.shape[0]
normal_mean_array = np.array(normal_mean_array)
normal_mean_array = np.reshape(normal_mean_array, (-1, 1))
distance = abs(predictions - normal_mean)


# Step 4: search for the best threshold (the one which maximizes the model's MCC)
# Finds a threshold the limits the distance that a data point prediction must have from the normal data prediction mean 
# in order to be considered normal. Data points which predictions distances lie outside that threshold are said to be anomalous.
best_threshold = -1
best_mcc = -1

interval = np.linspace(0.0, 0.5, num=100)
for threshold in interval:
    
    predictions = np.zeros_like(distance)

    predictions[distance > threshold] = 1

    mcc = metrics.matthews_corrcoef(labels, predictions)

    if mcc > best_mcc:
        best_mcc = mcc
        best_threshold = threshold


print('\n\nFound threshold: ', best_threshold)



# Step 5: save found threshold
threshold = {
    'th': best_threshold,
    'nmean': normal_mean 
}

from pickle import dump
_file = open('threshold.pkl', 'wb')
dump(threshold, _file)
_file.close()








# #
# # 3.2, Cacular a diferença entre o que foi previsto pelo modelo e o valor real observado.
# #      Espera-se que o erro seja maior nos segundos em que houve algum ataque.

# erro_abs_bytes_count   = abs(previsao[:, 0] - dia_2_y[:, 0])
# erro_abs_dst_ip_ent    = abs(previsao[:, 1] - dia_2_y[:, 1])
# erro_abs_dst_port_ent  = abs(previsao[:, 2] - dia_2_y[:, 2])
# erro_abs_src_ip_ent    = abs(previsao[:, 3] - dia_2_y[:, 3])
# erro_abs_src_port_ent  = abs(previsao[:, 4] - dia_2_y[:, 4])
# erro_abs_packets_count = abs(previsao[:, 5] - dia_2_y[:, 5])

# #erro_abs_geral: vetor de 86380 posicoes, onde cada posicao representa
# # o erro absoluto da previsao do modelo para aquele segundo.
# erro_abs_geral = erro_abs_dst_ip_ent + erro_abs_dst_port_ent + \
#                 erro_abs_src_ip_ent + erro_abs_src_port_ent + \
#                 erro_abs_bytes_count + erro_abs_packets_count  # shape: (86395, 1)



# # 3.3, encontrar um numero (threshold) que representa o maximo que o valor real pode se diferenciar
# #      do valor previsto para que o segundo ainda seja considerado como normal. Os segundos cujos
# #      valores reais estejam fora desse threshold são considerados anomalos.
# #
# melhor_f1 = -1
# melhor_threshold = -1

# # O thresold ideal sera aquele garanta o maior numero de acertos para as previsoes do modelo (melhor f1 score).
# # I é o intervalo no qual vamos procurar o threshold ideal.
# I = np.linspace(0.0, 10.0, num=100)
# for threshold in I:
#     # Ans é um array de 86380 posicoes que representa a resposta do modelo.
#     # obs: lembrar que o modelo começa a prever a partir do vigesimo segundo do dia por isso são apenas 86380 posicoes.
#     # Se ans[i] = 0, segundo o modelo, o i-esimo segundo é normal.
#     # Se ans[i] = 1, segundo o modelo, o i-esimo segundo é anomalo.
#     ans = np.zeros_like(erro_abs_geral)

#     # Os segundo nos quais o erro é maior que o threshold sao anomalos:
#     ans[erro_abs_geral > threshold] = 1

#     # Calcula o f1 score considerando as respostas atuais do modelo e as respostas corretas:
#     f1 = get_f1(ans, dia_2_rotulo)

#     # Se esse f1 score for melhor que o melhor f1 ja calculado vamos salvar o threshold atual e esse novo f1.
#     if f1 > melhor_f1:
#         melhor_f1 = f1
#         melhor_threshold = threshold


# print('\n\nThreshold: ', melhor_threshold)

# #
# ## Passo 4, Testar o modelo em um novo dia COM anomalias usando o threashold encontrado:
# #

# #
# # 4.1, Dia 1
# # 

# previsao = model.predict(dia_1_x) # shape: (86395, 6)


# erro_abs_bytes_count   = abs(previsao[:, 0] - dia_1_y[:, 0])
# erro_abs_dst_ip_ent    = abs(previsao[:, 1] - dia_1_y[:, 1])
# erro_abs_dst_port_ent  = abs(previsao[:, 2] - dia_1_y[:, 2])
# erro_abs_src_ip_ent    = abs(previsao[:, 3] - dia_1_y[:, 3])
# erro_abs_src_port_ent  = abs(previsao[:, 4] - dia_1_y[:, 4])
# erro_abs_packets_count = abs(previsao[:, 5] - dia_1_y[:, 5])

# erro_abs_geral = erro_abs_dst_ip_ent + erro_abs_dst_port_ent + \
#                 erro_abs_src_ip_ent + erro_abs_src_port_ent + \
#                 erro_abs_bytes_count + erro_abs_packets_count  # shape: (86399, 1)




# ans = np.zeros_like(erro_abs_geral)
# ans[erro_abs_geral > melhor_threshold] = 1

# precision, recall, f1 = get_f1(ans, dia_1_rotulo, all_metrics=True)
# print('\n\nDay 1 Metrics:')
# print("Confusion matrix: ")
# tn, fp, fn, tp = confusion_matrix(dia_1_rotulo, ans).ravel()
# print(np.array([[tp, fp], [fn, tn]]))
# print('Precision: ', precision)
# print('Recall: ', recall)
# print('F1: ', f1)