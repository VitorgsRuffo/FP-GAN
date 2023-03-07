import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pickle import dump, load


"""
Função que importa os dados ja preprocessados utilizando a abordagem ORION (6 features).
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
    scaler = MinMaxScaler((-1, 1)).fit(regular_day[:, 0:7]) 
    regular_day = scaler.transform(regular_day[:, 0:7])

    #saving scaler to disk so that it can be later used for scaling testing data...
    _file = open('normal_data_scaler.pkl', 'wb')
    dump(scaler, _file)
    _file.close()

    #returning data...
    return regular_day, scaler


#_, _ = import_gan_training_data()


def import_gan_testing_data(dataset='orion'):

    if dataset == 'orion':
        filename = r'../data/orion/'
        anomalous_day = np.array(pd.read_csv(filename+'171218_portscan_ddos_preprocessed.csv'))
        #anomalous_day = np.array(pd.read_csv(filename+'051218_portscan_preprocessed.csv'))

    labels = anomalous_day[:, 6]

    #adding timestamp to data...
    #( timestamp,bytes,dst_ip_entropy,dst_port_entropy,src_ip_entropy,src_port_entropy,packets,label )
    timestamp = [i for i in range(0,86400)]
    timestamp = np.array(timestamp)
    timestamp = np.reshape(timestamp, (86400, 1))
    anomalous_day = np.concatenate((timestamp, anomalous_day), axis=1)


    #scaling data...
    _file = open('normal_data_scaler.pkl', 'rb')
    scaler = load(_file) 
    anomalous_day = scaler.transform(anomalous_day[:, 0:7])
 

    #returning data...
    return anomalous_day, labels
