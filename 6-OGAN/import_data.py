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
        filename = r'../data/orion/'
        regular_day = np.array(pd.read_csv(filename+'051218_preprocessed.csv'))
    


    #timestamp, bytes, dst_ip_entropy, dst_port_entropy, src_ip_entropy, src_port_entropy, packets, label
    #adding timestamp to data...
    #this will hopefully help GAN generator understand how the 6 features are distribuited through time.
    timestamp = [i for i in range(0,86400)]
    timestamp = np.array(timestamp)
    timestamp = np.reshape(timestamp, (86400, 1))
    regular_day = np.concatenate((timestamp, regular_day), axis=1)


    regular_day = np.concatenate((regular_day[:, 0:1], regular_day[:, 1:2]), axis=1) ##change 2


    #scaling data...
    #scaler = StandardScaler().fit(regular_day[:, 0:6])
    scaler = MinMaxScaler((-1, 1)).fit(regular_day[:, 0:2]) 
    regular_day = scaler.transform(regular_day[:, 0:2])

    

    #returning data...
    return regular_day, scaler


regular_day, scaler = import_gan_training_data('orion')

print(regular_day)