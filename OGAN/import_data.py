import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


"""
Funções que importam os dados ja preprocessados utilizando a abordagem ORION (6 features).
"""
def import_gan_training_data(dataset='orion'):

    if dataset == 'orion':
        filename = r'../data/orion/'
        regular_day = np.array(pd.read_csv(filename+'051218_preprocessed.csv'))
        

    #adding timestamp to data...
    #( timestamp,bytes,dst_ip_entropy,dst_port_entropy,src_ip_entropy,src_port_entropy,packets,label )
    #this will hopefully help GAN generator understand how the 6 features are distribuited through time.
    timestamp = [i for i in range(0,86400)]
    timestamp = np.array(timestamp)
    timestamp = np.reshape(timestamp, (86400, 1))
    regular_day = np.concatenate((timestamp, regular_day), axis=1)



    #scaling data...
    #scaler = StandardScaler().fit(regular_day[:, 0:6])
    scaler = MinMaxScaler((-1, 1)).fit(regular_day[:, 0:7])
    regular_day = scaler.transform(regular_day[:, 0:7])
 

    #returning data...
    return regular_day, scaler


def import_gan_testing_data(dataset='orion'):

    if dataset == 'orion':
        filename = r'../data/orion/'
        anomalous_day_2 = np.array(pd.read_csv(filename+'171218_portscan_ddos_preprocessed.csv'))

    labels = anomalous_day_2[:, 7]

    #adding timestamp to data...
    #( timestamp,bytes,dst_ip_entropy,dst_port_entropy,src_ip_entropy,src_port_entropy,packets,label )
    timestamp = [i for i in range(0,86400)]
    timestamp = np.array(timestamp)
    timestamp = np.reshape(timestamp, (86400, 1))
    anomalous_day_2 = np.concatenate((timestamp, anomalous_day_2), axis=1)


    #scaling data...
    #scaler = StandardScaler().fit(anomalous_day_2[:, 0:6])
    scaler = MinMaxScaler((-1, 1)).fit(anomalous_day_2[:, 0:7])
    anomalous_day_2 = scaler.transform(anomalous_day_2[:, 0:7])
 

    #returning data...
    return anomalous_day_2, labels, scaler