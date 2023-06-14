import pandas as pd
import numpy as np
import sys

if len(sys.argv) < 8:
    print("Please.\nSpecify the required parameters.\n")
    quit(1)

path = sys.argv[1]
day_type = int(sys.argv[2])
day_name =  sys.argv[3]
interval00 = sys.argv[4]
interval01 = sys.argv[5]
interval10 = sys.argv[6]
interval11 = sys.argv[7]


bits_count = np.loadtxt(f'{path}/bytes1.txt').reshape(-1, 1)   # shape: (86400, 1)
dst_ip_ent = np.loadtxt(f'{path}/EntropiaDstIP1.txt').reshape(-1, 1)
dst_port_ent = np.loadtxt(f'{path}/EntropiaDstPort1.txt').reshape(-1, 1)
src_ip_ent = np.loadtxt(f'{path}/EntropiaScrIP1.txt').reshape(-1, 1)
src_port_ent = np.loadtxt(f'{path}/EntropiaSrcPort1.txt').reshape(-1, 1)
packets_count = np.loadtxt(f'{path}/packets1.txt').reshape(-1, 1)
labels = np.zeros((86400, 1))

data_matrix = np.column_stack(
    (bits_count, dst_ip_ent, dst_port_ent, src_ip_ent, src_port_ent, packets_count, labels))

columns = \
    ['bits','dst_ip_entropy','dst_port_entropy','src_ip_entropy','src_port_entropy','packets', 'label']

data_frame = pd.DataFrame(data_matrix, columns=columns, index=None)


def time_to_seconds(time: str):
    h, m, s = time.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


def get_labels(anomalous_intervals):
    labels = np.zeros(86400, dtype=np.int8)
    anomalous_intervals_in_seconds = []
    for anomalous_interval in anomalous_intervals:
        start = time_to_seconds(anomalous_interval[0])
        finish = time_to_seconds(anomalous_interval[1])
        anomalous_intervals_in_seconds.append((start, finish))

    # Para cada intervalo anomalo...
    for anomalous_interval in anomalous_intervals_in_seconds:
        # os segundos dentro desse intervalo sao marcados como anomalos ( rótulo == 1 ).
        for i in range(anomalous_interval[0], anomalous_interval[1] + 1):
            labels[i] = 1
    return labels


if day_type != 0:
    labels = get_labels([(interval00, interval01), (interval10, interval11)])
    data_frame['label'] = labels


#save new .csv files
data_frame.to_csv(f"./{path}/{day_name}.csv", index=False, header=True)
