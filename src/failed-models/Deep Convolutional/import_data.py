import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pickle import dump, load


def import_orion_normal_data():

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

    regular_day = np.reshape(regular_day, (regular_day.shape[0], 1, regular_day.shape[1])) ## DCGAN change
    #regular_day = np.reshape(regular_day, (regular_day.shape[0], regular_day.shape[1], 1)) ## DCGAN change

    #returning data...
    return regular_day, scaler


def import_orion_normal_windowed_data(network_type='dnn', window_size=5):

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

    #reformatting data (sliding window format) 
    #shape: [samples (windows), time steps (window size), features (window element size)]
    regular_day_x, regular_day_y = [], []
    dataset_len = len(regular_day)
    for i in range(dataset_len-window_size - 1):
        aux = regular_day[i:(i+window_size), :]
        regular_day_x.append(aux)
        regular_day_y.append(regular_day[i + window_size, :])
    regular_day_x = np.array(regular_day_x)
    regular_day_y = np.array(regular_day_y)

    if network_type == 'dnn': #adjust format a little bit.
        #shape: [samples (windows), time steps (window size) * features (window element size)]
        #window distinction are maintained. window elements are all merged together.
        regular_day_x = np.reshape(regular_day_x, (regular_day_x.shape[0], -1)) 


    #returning data...
    return regular_day_x, regular_day_y

#_, _ = import_gan_training_data()
















def import_orion_anomalous_data(day=1, portscan=False):

    filename = r'../data/orion/'
    anomalous_day = None
    if day == 1:
        anomalous_day = np.array(pd.read_csv(filename+'051218_ddos_portscan_preprocessed.csv'))
    
    else:
        anomalous_day = np.array(pd.read_csv(filename+'171218_portscan_ddos_preprocessed.csv'))

    labels = anomalous_day[:, 6]


    #adding timestamp to data...
    #( timestamp,bytes,dst_ip_entropy,dst_port_entropy,src_ip_entropy,src_port_entropy,packets,label )
    timestamp = [i for i in range(0,86400)]
    timestamp = np.array(timestamp)
    timestamp = np.reshape(timestamp, (86400, 1))
    anomalous_day = np.concatenate((timestamp, anomalous_day), axis=1)


    #scaling data...
    try:
        _file = open('normal_data_scaler.pkl', 'rb')
    except:
        return None, None
    scaler = load(_file) 
    anomalous_day = scaler.transform(anomalous_day[:, 0:7])


    if not portscan:
        if day == 1:
            # 1st interval (ddos): 36900-41400  
            # 2nd interval (portscan): 48300-52500
            anomalous_day = np.concatenate((anomalous_day[0:48300, :], anomalous_day[52501:, :]), axis=0)
            labels = np.concatenate((labels[0:48300], labels[52501:]), axis=0)
        else:
            # 1st interval (portscan): 35100-40200
            # 2nd interval (ddos): 63420-68100
            anomalous_day = np.concatenate((anomalous_day[0:35100, :], anomalous_day[40201:, :]), axis=0)
            labels = np.concatenate((labels[0:35100], labels[40201:]), axis=0)


    anomalous_day = np.reshape(anomalous_day, (anomalous_day.shape[0], 1, anomalous_day.shape[1])) ## DCGAN change
    #anomalous_day = np.reshape(anomalous_day, (anomalous_day.shape[0], anomalous_day.shape[1], 1)) ## DCGAN change

    #returning data...
    return anomalous_day, labels















def import_orion_anomalous_windowed_data(network_type='dnn', window_size=5):

    
    filename = r'../data/orion/'
    anomalous_day_1 = np.array(pd.read_csv(filename+'051218_ddos_portscan_preprocessed.csv'))
    anomalous_day_2 = np.array(pd.read_csv(filename+'171218_portscan_ddos_preprocessed.csv'))

    labels_1 = anomalous_day_1[:, 6]
    labels_2 = anomalous_day_2[:, 6]


    #adding timestamp to data...
    #( timestamp,bytes,dst_ip_entropy,dst_port_entropy,src_ip_entropy,src_port_entropy,packets,label )
    timestamp = [i for i in range(0,86400)]
    timestamp = np.array(timestamp)
    timestamp = np.reshape(timestamp, (86400, 1))
    anomalous_day_1 = np.concatenate((timestamp, anomalous_day_1), axis=1)
    anomalous_day_2 = np.concatenate((timestamp, anomalous_day_2), axis=1)


    #scaling data...
    _file = open('normal_data_scaler.pkl', 'rb')
    scaler = load(_file) 
    anomalous_day_1 = scaler.transform(anomalous_day_1[:, 0:7])
    anomalous_day_2 = scaler.transform(anomalous_day_2[:, 0:7])


    #reformatting data (sliding window format) 
    #shape: [samples (windows), time steps (window size), features (window element size)]
    anomalous_day_1_x, anomalous_day_1_y = [], []
    dataset_len = len(anomalous_day_1)
    for i in range(dataset_len-window_size - 1):
        aux = anomalous_day_1[i:(i+window_size), :]
        anomalous_day_1_x.append(aux)
        anomalous_day_1_y.append(anomalous_day_1[i + window_size, :])
    anomalous_day_1_x = np.array(anomalous_day_1_x)
    anomalous_day_1_y = np.array(anomalous_day_1_y)

    if network_type == 'dnn': #adjust format a little bit.
        #shape: [samples (windows), time steps (window size) * features (window element size)]
        #window distinction are maintained. window elements are all merged together.
        anomalous_day_1_x = np.reshape(anomalous_day_1_x, (anomalous_day_1_x.shape[0], -1)) 

    anomalous_day_2_x, anomalous_day_2_y = [], []
    dataset_len = len(anomalous_day_2)
    for i in range(dataset_len-window_size - 1):
        aux = anomalous_day_2[i:(i+window_size), :]
        anomalous_day_2_x.append(aux)
        anomalous_day_2_y.append(anomalous_day_2[i + window_size, :])
    anomalous_day_2_x = np.array(anomalous_day_2_x)
    anomalous_day_2_y = np.array(anomalous_day_2_y)

    if network_type == 'dnn': #adjust format a little bit.
        #shape: [samples (windows), time steps (window size) * features (window element size)]
        #window distinction are maintained. window elements are all merged together.
        anomalous_day_2_x = np.reshape(anomalous_day_2_x, (anomalous_day_2_x.shape[0], -1)) 



    #returning data...
    return anomalous_day_1_x, anomalous_day_1_y, labels_1, anomalous_day_2_x, anomalous_day_2_y, labels_2

