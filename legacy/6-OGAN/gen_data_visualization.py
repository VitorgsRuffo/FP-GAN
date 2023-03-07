import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import sys

if len(sys.argv) <= 1:
    print("No input file provided.\n")
    quit(1)


def time_to_seconds(time: str):
    h, m, s = time.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


day_file_path = sys.argv[1]

day = sys.argv[2]

data_frame = pd.read_csv(day_file_path)
data_frame = data_frame.sort_values('timestamp')


# features_names = ['bytes', 'packets', 'src_ip_entropy', 'src_port_entropy', 
#                   'dst_ip_entropy', 'dst_port_entropy']

features_names = ['bytes'] ##change 3

titles = {
    'bytes': 'Total de bits',
    'packets': 'Total de pacotes',
    'src_ip_entropy': 'Entropia de IP de origem',
    'src_port_entropy': 'Entropia de porta de origem',
    'dst_ip_entropy': 'Entropia de IP de destino',
    'dst_port_entropy': 'Entropia de porta de destino'
}

attack_intervals = {
    '1': [],
    '2': [('10:15:00', '11:30:00'), ('13:25:00', '14:35:00')],
    '3': [('09:45:00', '11:10:00'), ('17:37:00', '18:55:00')]
}

mpl.rcParams['lines.linewidth'] = 0.5
figure, plots = plt.subplots(3, 2, figsize=(10.80,7.20))
#plt.suptitle(day_file_path.split("/")[-1], fontsize=14)

lin_space = np.linspace(0, 24, data_frame.shape[0])

for k in range(0, 1):
    i = k // 2 # mapping array index to matrix indices. (obs: 2 == matrix line length)
    j = k % 2
    features = np.array(data_frame[features_names[k]])
    plots[i][j].set_title(titles[features_names[k]])
    plots[i][j].set(xlabel="Tempo (hora)",ylabel=titles[features_names[k]])
    plots[i][j].step(lin_space, features, color='darkgreen')
    plots[i][j].set_facecolor("lightgray")
    plots[i][j].margins(x=0)
    plots[i][j].set_xticks(np.arange(0, 25, 2))

    #attack_intervals: lista de tuplas, onde cada tupla representa um intervalo de ataque (e.g., [('09:45:00', '11:10:00'), ('17:37:00', '18:55:00')]).
    
    for attack_interval in attack_intervals[day]:
        start, end = \
            time_to_seconds(attack_interval[0]), time_to_seconds(attack_interval[1])
        start, end = lin_space[start], lin_space[end]
        plots[i][j].axvspan(start, end, label="ATTACK", color='r', alpha=0.5)


    if k == 0:
        pass
        #plots[i][j].set_ylim([0, x])
        plots[i][j].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    elif k == 1:
        pass
        #plots[i][j].set_ylim([0, y])
        plots[i][j].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    else:
        plots[i][j].set_ylim([-1, 7])

    n = 0

output_path = day_file_path.split(".csv")[0] + str(".jpg")
plt.tight_layout()
plt.savefig(output_path, format='jpg', dpi=800)
