import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pickle import dump, load



def import_orion_normal_windowed_data(dataset=1, window_size=5):
    regular_day = None
    if dataset == 1:
        regular_day = np.array(pd.read_csv('../data/orion/o1_access_2020_gustavo/051218_60h6sw_c1_ht5_it0_V2_csv/051218.csv'))
    else:
        regular_day = np.array(pd.read_csv('../data/orion/o2_globecom_2022/071218_140h6sw_c1_ht5_it0_V2_csv/071218.csv'))


    #scaling data...
    #(bits,dst_ip_entropy,dst_port_entropy,src_ip_entropy,src_port_entropy,packets,label)
    scaler = StandardScaler().fit(regular_day[:, 0:6])
    regular_day = scaler.transform(regular_day[:, 0:6])


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


    #returning data...
    return regular_day_x, regular_day_y

#_, _ = import_orion_normal_windowed_data()









def import_orion_anomalous_windowed_data(dataset=1, day=1, portscan=False, window_size=5):
    anomalous_day = None
    if dataset == 1: #access 2020
        if day == 1:
            anomalous_day = np.array(pd.read_csv('../data/orion/o1_access_2020_gustavo/051218_60h6sw_c1_ht5_it0_V2_csv_ddos_portscan/051218_ddos_portscan.csv'))
        else:
            anomalous_day = np.array(pd.read_csv('../data/orion/o1_access_2020_gustavo/171218_60h6sw_c1_ht5_it0_V2_csv_portscan_ddos/171218_portscan_ddos.csv'))
       
    else: #globecom 2022
        if day == 1:
            anomalous_day = np.array(pd.read_csv('../data/orion/o2_globecom_2022/071218_140h6sw_c1_ht5_it0_V2_csv_ddos_portscan/071218_ddos_portscan.csv'))
        else:
            anomalous_day = np.array(pd.read_csv('../data/orion/o2_globecom_2022/071218_140h6sw_c1_ht5_it0_V2_csv_portscan_ddos/071218_portscan_ddos.csv'))

    #( bits,dst_ip_entropy,dst_port_entropy,src_ip_entropy,src_port_entropy,packets,label )
    labels = anomalous_day[:, 6]



    #scaling data...
    try:
        _file = open('normal_data_scaler.pkl', 'rb')
    except:
        return None, None
    scaler = load(_file) 
    anomalous_day = scaler.transform(anomalous_day[:, 0:6])

    #conditionally removing portscan...
    if not portscan:
        if dataset == 1:  #access 2020
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
        
        else:  #globecom 2022
            if day == 1:
                # 1st interval (ddos): 33300-37800
                # 2nd interval (portscan): 49500-53400
                anomalous_day = np.concatenate((anomalous_day[0:49500, :], anomalous_day[53401:, :]), axis=0)
                labels = np.concatenate((labels[0:49500], labels[53401:]), axis=0)
            else:
                # 1st interval (portscan): 40800-45600
                # 2nd interval (ddos): 59700-64200
                anomalous_day = np.concatenate((anomalous_day[0:40800, :], anomalous_day[45601:, :]), axis=0)
                labels = np.concatenate((labels[0:40800], labels[45601:]), axis=0)


    #reformatting data (sliding window format) 
    #shape: [samples (windows), time steps (window size), features (window element size)]
    anomalous_day_x, anomalous_day_y = [], []
    dataset_len = len(anomalous_day)
    for i in range(dataset_len-window_size):
        aux = anomalous_day[i:(i+window_size), :]
        anomalous_day_x.append(aux)
        anomalous_day_y.append(anomalous_day[i + window_size, :])
    anomalous_day_x = np.array(anomalous_day_x)
    anomalous_day_y = np.array(anomalous_day_y)
    labels = labels[window_size:]

    #returning data...
    return anomalous_day_x, anomalous_day_y, labels 

#_, _, _ = import_orion_anomalous_windowed_data()
