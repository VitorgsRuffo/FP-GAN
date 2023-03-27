import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt




"""
Converte uma string representando o formato de tempo 'hh:mm:ss'
para a quantidade de segundos correspondente (à partir de 00:00:00).
"""
def time_to_seconds(time: str):
    h, m, s = time.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


"""
Plota os graficos que mostram a previsão do modelo vs os dados reais. 
Assim, podemos vizualizar se a caracterização do modelo esta boa. Isto é, se ele aprendeu o comportamento 
do tráfego e esta conseguindo realizar previsões proximas da realidade.
"""
def plot_prediction_real_graph(graph_name: str, save_path: str, predicted_data, real_data, attack_intervals: list, window_size=20):
    features_names = ['bytes', 'dst_ip_entropy', 'dst_port_entropy',
                    'src_ip_entropy', 'src_port_entropy', 'packets']
    mpl.rcParams['lines.linewidth'] = 0.5
    #mpl.style.use('seaborn') #['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']
    figure, plots = plt.subplots(3, 2, figsize=(10.80,7.20))
    plt.suptitle(graph_name, fontsize=14)

    lin_space = np.linspace(0, 24, real_data.shape[0])
    for k in range(0, 6):
        i = k // 2 # mapping array index to matrix indices. (obs: 2 == matrix line length)
        j = k % 2
        plots[i][j].set_title(features_names[k])
        plots[i][j].step(lin_space, real_data[:, k], label='REAL', color='green')
        plots[i][j].step(lin_space, predicted_data[:, k], label='PREDICTED', color='orange')
        plots[i][j].set_xticks(np.arange(0, 25, 2))
        plots[i][j].margins(x=0)
        plots[i][j].legend(fontsize='xx-small')

        # mse = round(mean_squared_error(real_data[:, k], predicted_data[:, k]), 6)
        # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # plots[i][j].text(0.05, 0.90, f"MSE = {mse}", 
        #     transform=plots[i][j].transAxes, fontsize=7, bbox=props) 
        
        for attack_interval in attack_intervals:
            start, end = \
                time_to_seconds(attack_interval[0]), time_to_seconds(attack_interval[1])
            start, end = lin_space[start], lin_space[end]
            plots[i][j].axvspan(start, end, label="ATTACK", color='r', alpha=0.5)

        if k == 0:
            #plots[i][j].set_ylim([0, x])
            plots[i][j].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        elif k == 5:
            #plots[i][j].set_ylim([0, y])
            plots[i][j].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        else:
            plots[i][j].set_ylim([-10, 10])

    plt.tight_layout()
    plt.savefig(save_path, format='jpg', dpi=800)
    plt.close()

