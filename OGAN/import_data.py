import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


"""
Função que importa os dados ja preprocessados utilizando a abordagem ORION (6 features).
"""
def import_gan_training_data(dataset='orion'):

    if dataset == 'orion':
        filename = r'./data/orion/'
        regular_day = np.array(pd.read_csv(filename+'051218_preprocessed.csv'))
        anomalous_day_1 = np.array(pd.read_csv(filename+'051218_portscan_preprocessed.csv'))
        #anomalous_day_2 = np.array(pd.read_csv(filename+'171218_portscan_ddos_preprocessed.csv'))
    

    #adding timestamp (min) to data...
    #this will hopefully help GAN generator understand how the 6 features are distribuited through time.
    timestamp = list()
    for i in range(0,1440):
        timestamp.extend([i] * 60)
    timestamp = np.array(timestamp)
    timestamp = np.reshape(timestamp, (86400, 1))

    regular_day = np.concatenate((timestamp, regular_day), axis=1)
    anomalous_day_1 = np.concatenate((timestamp, anomalous_day_1), axis=1)


    #scaling data...
    #scaler = StandardScaler().fit(regular_day[:, 0:6])
    scaler = MinMaxScaler((-1, 1)).fit(regular_day[:, 0:7])
    #scaler = MinMaxScaler().fit(regular_day[:, 0:7])
    regular_day[:, 0:7] = scaler.transform(regular_day[:, 0:7])
    anomalous_day_1[:, 0:7] = scaler.transform(anomalous_day_1[:, 0:7])
    #anomalous_day_2[:, 0:7] = scaler.transform(anomalous_day_2[:, 0:7])




    #returning data...
    return regular_day, anomalous_day_1, scaler


# regular_day, anomalous_day_1, scaler = import_gan_training_data('orion')

# print(regular_day)

# generated_data = np.array(pd.read_csv('model2/generated_regular_day.csv'))

# generated_data = scaler.inverse_transform(generated_data) #scaling data to original range


# columns = \
#     ["timestamp (min)", "bytes", "dst_ip_entropy",  "dst_port_entropy", "src_ip_entropy", "src_port_entropy", "packets"]


# data_frame = pd.DataFrame(generated_data, columns=columns, index=None)

# import os
# data_frame.to_csv(f'./generated_regular_day_scaled.csv', index=False)
# os.system(f'python3 gen_data_visualization.py ./generated_regular_day_scaled.csv 1')
