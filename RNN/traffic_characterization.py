# Módulo de caracterização: 
#(modelo Multi-Layer Perceptron (MLP) de regressão (Machine Learning supervisionado))
#
# O sistema aprendera o perfil de comportamento normal do trafego da rede (baseline) 
# utilizando um dia sem ataques, e, sera definido um limite (threshold) para a variação do comportamento
# da rede.

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import GRU, LSTM, Dense, Dropout, LeakyReLU, Input
from sklearn.metrics import f1_score
from tensorflow.keras.optimizers import Adam
import sys
from import_data import import_orion_normal_windowed_data, import_orion_anomalous_windowed_data
from utils import plot_prediction_real_graph

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score


#
## Passo 1, importar os dados:
#

# O modelo recebe uma janela de window_size segundos como entrada 
# e preve o segundo imediatamente após essa janela. 
window_size = 10
portscan = False

no_of_features = 6 #numero de colunas (caracteristicas ou features) dos dados de entrada.

normal_data_x, normal_data_y, scaler = import_orion_normal_windowed_data(dataset=1, window_size=window_size)


anomalous_data_1_x, anomalous_data_1_y,\
anomalous_data_1_labels, _ = import_orion_anomalous_windowed_data(dataset=1, day=1, portscan=portscan, window_size=window_size)



#
## Passo 2: criar um modelo de regressão: 
#


# 2.1 configurando o modelo:

def build_model(window_size, no_of_features):
    model = Sequential()
    model.add(Input(shape=(window_size, no_of_features)))
    model.add(GRU(32)) 
    #model.add(LSTM(32)) 
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))
    model.add(Dense(24))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))
    model.add(Dense(no_of_features, activation=None))
    return model


model = build_model(window_size, no_of_features)
learning_rate = 0.001
batch_size = 64
epochs = 20
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='mse', optimizer=optimizer) 
print("\nOrganização do modelo:")
model.summary()


# 2.2 treinando o modelo:
# hist = model.fit(
#     x=normal_data_x,
#     y=normal_data_y,
#     #dados de validação são importantes para verificar a performance do modelo
#     #para dados desconhecidos ao longo do treinamento. São uteis para 
#     #ajustar o hiperparametros! 
#     #validation_data=(x, y), 
#     batch_size=batch_size,
#     epochs=epochs
# )

# #plotting loss
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import locale
# plt.rcParams['axes.formatter.use_locale'] = True

# plt.plot(hist.history['loss'], color='#379237')


# locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
# import matplotlib.ticker as tkr
# def func(x, pos):  # formatter function takes tick label and tick position
#     return locale.format_string("%.2f", x)
# axis_format = tkr.FuncFormatter(func)  # make formatter

# mpl.rcParams['lines.linewidth'] = 1

# plt.xlabel('Época')
# plt.ylabel('Erro')
# ax = plt.gca()

# from matplotlib.ticker import MaxNLocator
# ax.xaxis.set_major_locator(MaxNLocator(integer=True)) 
# ax.yaxis.set_major_formatter(axis_format) #when using ','

# import numpy as np
# plt.yticks(np.arange(0.30, 0.50, step=0.02))

# plt.margins(x=0)
# plt.savefig(f"./loss.png", dpi=150)
# plt.close()


# # 2.3 salvando o modelo treinado no disco...
# model.save('./model')

model = load_model('./model')

#check baseline learning:
prediction = model.predict(normal_data_x) # shape: (86380, 6)
plot_prediction_real_graph("Prediction-real graph:\n051218", 
                            "./051218_prediction_real_graph.jpg", '1',
                            prediction, normal_data_y, scaler, window_size)


#
## Passo 3, encontrar o threshold que garanta o melhor f1 score para o modelo:
#

# 3.1, Colocando o modelo para prever um dia COM ataque. Já que o modelo
# só foi treinado para prever segundos sem ataque, ele provavelmente
# vai errar a previsão quando encontrar ataques. Em outras palavras, considerando
# a presença de anomalias, a diferença entre os atributos do segundo real e os
# do segundo previsto vai ser consideravelmente grande.
prediction = model.predict(anomalous_data_1_x) # shape: (86380, 6)


#
# 3.1.1 Plotando os graficos que mostram a previsão do modelo vs os dados reais. 
# Assim, podemos vizualizar se a caracterização do modelo esta boa. Isto é, se ele aprendeu o comportamento 
# do tráfego e esta conseguindo realizar previsões proximas da realidade.


plot_prediction_real_graph("Prediction-real graph:\n051218_ddos_portscan", 
                           "./051218_ddos_portscan_prediction_real_graph.jpg", '2',
                           prediction, anomalous_data_1_y, scaler, window_size)




#
# 3.2, Cacular a diferença entre o que foi previsto pelo modelo e o valor real observado.
#      Espera-se que o erro seja maior nos segundos em que houve algum ataque.

bytes_count_abs_error   = abs(prediction[:, 0] - anomalous_data_1_y[:, 0])
dst_ip_ent_abs_error    = abs(prediction[:, 1] - anomalous_data_1_y[:, 1])
dst_port_ent_abs_error  = abs(prediction[:, 2] - anomalous_data_1_y[:, 2])
src_ip_ent_abs_error    = abs(prediction[:, 3] - anomalous_data_1_y[:, 3])
src_port_ent_abs_error  = abs(prediction[:, 4] - anomalous_data_1_y[:, 4])
packets_count_abs_error = abs(prediction[:, 5] - anomalous_data_1_y[:, 5])


#vetor de 86380 posicoes, onde cada posicao representa
# o erro absoluto da previsao do modelo para aquele segundo.
abs_error = dst_ip_ent_abs_error + dst_port_ent_abs_error + \
            src_ip_ent_abs_error + src_port_ent_abs_error + \
            bytes_count_abs_error + packets_count_abs_error 




# 3.3, encontrar um numero (threshold) que representa o maximo que o valor real pode se diferenciar
#      do valor previsto para que o segundo ainda seja considerado como normal. Os segundos cujos
#      diferenças estejam fora desse threshold são considerados anomalos.
#
best_mcc = -1
best_threshold = -1

# O thresold ideal sera aquele garanta o maior numero de acertos para as previsoes do modelo (melhor mcc score).
# I é o intervalo no qual vamos procurar o threshold ideal.
interval = np.linspace(0.0, 30.0, num=200)
for threshold in interval:
    # predicted_labels é um array de 86380 posicoes que representa a resposta do modelo.
    # obs: lembrar que o modelo começa a prever a partir do vigesimo segundo do dia por isso são apenas 86380 posicoes.
    # Se predicted_labels[i] = 0, segundo o modelo, o i-esimo segundo é normal.
    # Se predicted_labels[i] = 1, segundo o modelo, o i-esimo segundo é anomalo.
    predicted_labels = np.zeros_like(abs_error)

    # Os segundo nos quais o erro é maior que o threshold sao anomalos:
    predicted_labels[abs_error > threshold] = 1

    # Calcula o mcc considerando as respostas atuais do modelo e as respostas corretas:
    mcc = matthews_corrcoef(anomalous_data_1_labels, predicted_labels)

    # Se esse mcc for melhor que o melhor mcc ja calculado vamos salvar o threshold atual e esse novo mcc.
    if mcc > best_mcc:
        best_mcc = mcc
        best_threshold = threshold
print('\n\nBest threshold: ', best_threshold)


#3.4 salvando no disco o threshold encontrado ...
from pickle import dump
_file = open('threshold.pkl', 'wb')
dump(best_threshold, _file)
_file.close()
