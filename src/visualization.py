import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os
import sys
import locale

if len(sys.argv) <= 1:
    print("No input file provided.\n")
    quit(1)


def time_to_seconds(time: str):
    h, m, s = time.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)



plt.rcParams['axes.formatter.use_locale'] = True


day_file_path = sys.argv[1]

day = sys.argv[2]

normal_data_frame = np.array(pd.read_csv("051218_60h6sw_c1_ht5_it0_V2_csv/051218.csv"))
data_frame = np.array(pd.read_csv(day_file_path))

features_names = ['bits', 'packets', 'src_ip_entropy', 'src_port_entropy', 
                  'dst_ip_entropy', 'dst_port_entropy']



titles = {
    'bits': 'Total de bits',
    'packets': 'Total de pacotes',
    'src_ip_entropy': 'Entropia IP de origem',
    'src_port_entropy': 'Entropia de porta de origem',
    'dst_ip_entropy': 'Entropia de IP de destino',
    'dst_port_entropy': 'Entropia de porta de destino'
}

y_axis_labels = {
    'bits': 'bits',
    'packets': 'pacotes',
    'src_ip_entropy': 'H(IP de origem)',
    'src_port_entropy': 'H(porta de origem)',
    'dst_ip_entropy': 'H(IP de destino)',
    'dst_port_entropy': 'H(porta de destino)'
}

#attack_intervals = {
#    '1': [],
#    '2': [('10:15:00', '11:30:00'), ('13:25:00', '14:35:00')],
#    '3': [('09:45:00', '11:10:00'), ('17:37:00', '18:55:00')]
#}

attack_intervals = {
    '1': [],
    '2': [('10:15:00', '11:30:00')],
    '3': [('17:37:00', '18:55:00')]
}

#removing portscan
if day == '2':
    #data_frame = np.concatenate((data_frame[0:48300, :], data_frame[52501:, :]), axis=0)
    data_frame[48300:52501, :] = normal_data_frame[48300:52501, :]

elif day == '3':
    #data_frame = np.concatenate((data_frame[0:35100, :], data_frame[40201:, :]), axis=0)
    data_frame[35100:40201, :] = normal_data_frame[35100:40201, :]

data_frame = pd.DataFrame(data_frame, columns = ['bits','dst_ip_entropy','dst_port_entropy','src_ip_entropy','src_port_entropy','packets','label'])

mpl.rcParams['lines.linewidth'] = 0.5
figure, plots = plt.subplots(3, 2, figsize=(10.80,7.20))
#plt.suptitle(day_file_path.split("/")[-1], fontsize=14)



#y axis format configuration:
locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
import matplotlib.ticker as tkr
def func(x, pos):  # formatter function takes tick label and tick position
    return locale.format_string("%.1f", x)
y_format = tkr.FuncFormatter(func)  # make formatter


lin_space = np.linspace(0, 24, data_frame.shape[0])

for k in range(0, 6):
    i = k // 2 # mapping array index to matrix indices. (obs: 2 == matrix line length)
    j = k % 2
    features = np.array(data_frame[features_names[k]])
    plots[i][j].set_title(titles[features_names[k]])
    plots[i][j].set(xlabel="Tempo (hora)",ylabel=y_axis_labels[features_names[k]])
    plots[i][j].step(lin_space, features, color='darkgreen')
    #plots[i][j].set_facecolor("lightgray")
    plots[i][j].fill_between(
        lin_space, 
        features,
        -10,
        color= "g",
        alpha= 0.2)    
    plots[i][j].margins(x=0)
    plots[i][j].set_xticks(np.arange(0, 25, 2))

    #attack_intervals: lista de tuplas, onde cada tupla representa um intervalo de ataque (e.g., [('09:45:00', '11:10:00'), ('17:37:00', '18:55:00')]).
    
#    if day == '3':
#
#        start, end = \
#                time_to_seconds(attack_intervals[day][0][0]), time_to_seconds(attack_intervals[day][0][1])
#        portscan_duration = end-start
#
#        start, end = \
#                time_to_seconds(attack_intervals[day][1][0]), time_to_seconds(attack_intervals[day][1][1])
#        start = start - portscan_duration 
#        end = end - portscan_duration
#        
#        start, end = lin_space[start], lin_space[end]
#        plots[i][j].axvspan(start, end, label="ATTACK", color='r', alpha=0.5)
#        plots[i][j].step(lin_space[start:end], features[start:end], color='darkred')
#        plots[i][j].fill_between(
#            lin_space[start:end], 
#            features[start:end],
#            -10,
#            color= "r",
#            alpha= 0.2)
#    
#    else:
    for attack_interval in attack_intervals[day]:
        start, end = \
            time_to_seconds(attack_interval[0]), time_to_seconds(attack_interval[1])

        plots[i][j].step(lin_space[start:end], features[start:end], color='darkred')
        plots[i][j].fill_between(
        lin_space[start:end], 
        features[start:end],
        -10,
        color= "r",
        alpha= 0.2)  
#            start, end = lin_space[start], lin_space[end]
#            plots[i][j].axvspan(start, end, label="ATTACK", color='r', alpha=0.5)


    if k == 0:
        pass
        plots[i][j].set_ylim([0, 250000])
        plots[i][j].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    elif k == 1:
        pass
        plots[i][j].set_ylim([0, 1500])
        plots[i][j].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    else:
        plots[i][j].set_ylim([-1, 8])
        #plots[i][j].yaxis.set_major_formatter(FormatStrFormatter('%.1f')) #when using '.'
        plots[i][j].yaxis.set_major_formatter(y_format) #when using ','

    n = 0

output_path = day_file_path.split(".csv")[0] + str(".jpg")
plt.tight_layout()
plt.savefig(output_path, format='jpg', dpi=800)
