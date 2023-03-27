import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os
import sys
from import_data import import_orion_normal_data, import_cic_normal_data
from sklearn.metrics import mean_squared_error


if len(sys.argv) <= 1:
    print("No input file provided.\n")
    quit(1)


def time_to_seconds(time: str):
    h, m, s = time.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


day_file_path = sys.argv[1]
gen_data = pd.read_csv(day_file_path)
gen_data = gen_data.sort_values('timestamp')


dataset = sys.argv[2]
original_data = None
if dataset == 'orion1':
    original_data, scaler = import_orion_normal_data(dataset=1)

elif dataset == 'orion2':
    original_data, scaler = import_orion_normal_data(dataset=2)

else:
    pass

original_data = scaler.inverse_transform(original_data) #scaling data to original range
original_data = pd.DataFrame(original_data, columns = ['timestamp','bits','dst_ip_entropy','dst_port_entropy','src_ip_entropy','src_port_entropy','packets'])



features_names = ['bits', 'packets', 'src_ip_entropy', 'src_port_entropy', 
                  'dst_ip_entropy', 'dst_port_entropy']

titles = {
    'bits': 'Total de bits',
    'packets': 'Total de pacotes',
    'src_ip_entropy': 'E(IP de origem)',
    'src_port_entropy': 'E(porta de origem)',
    'dst_ip_entropy': 'E(IP de destino)',
    'dst_port_entropy': 'E(porta de destino)'
}


mpl.rcParams['lines.linewidth'] = 0.5
figure, plots = plt.subplots(3, 2, figsize=(10.80,7.20))
#plt.suptitle(day_file_path.split("/")[-1], fontsize=14)

lin_space = None
if dataset == 'cic':
    lin_space = np.linspace(9.5, 17.5, gen_data.shape[0])
else:
    lin_space = np.linspace(0, 24, gen_data.shape[0])

for k in range(0, 6):
    i = k // 2 # mapping array index to matrix indices. (obs: 2 == matrix line length)
    j = k % 2
    #plots[i][j].set_title(titles[features_names[k]])
    plots[i][j].set(xlabel="Tempo (hora)",ylabel=titles[features_names[k]])
    
    original_features = np.array(original_data[features_names[k]])
    plots[i][j].step(lin_space, original_features, color='darkgreen', label='Distribuição original')

    gen_features = np.array(gen_data[features_names[k]])
    plots[i][j].step(lin_space, gen_features, color='orange', label='Distribuição aprendida', alpha=0.65)
   
    plots[i][j].set_xticks(np.arange(0, 25, 2))

    plots[i][j].set_facecolor("lightgray")
    plots[i][j].margins(x=0)

    # mse = round(mean_squared_error(original_features, gen_features), 6)
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # plots[i][j].text(0.05, 0.90, f"MSE = {mse}", 
    #     transform=plots[i][j].transAxes, fontsize=7, bbox=props) 
        

    if k == 0:
        pass
        #plots[i][j].set_ylim([0, x])
        plots[i][j].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    elif k == 1:
        pass
        #plots[i][j].set_ylim([0, y])
        plots[i][j].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    else:
        plots[i][j].set_ylim([-1, 8.0])
        plots[i][j].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    n = 0

handles, labels = plots[2, 1].get_legend_handles_labels()
figure.legend(handles, labels, loc="lower left")

output_path = day_file_path.split(".csv")[0] + str(".jpg")
plt.tight_layout(pad=1)
figure.subplots_adjust(bottom=0.15)
plt.savefig(output_path, format='jpg', dpi=800)
