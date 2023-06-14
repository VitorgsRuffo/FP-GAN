import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import locale



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
def plot_prediction_real_graph(graph_name: str, save_path: str, day, predicted_data, real_data, scaler, window_size=20):
    features_names = ['bits', 'packets', 'src_ip_entropy', 'src_port_entropy', 
                  'dst_ip_entropy', 'dst_port_entropy']
    
    y_labels = {
    'bits': 'Total de bits',
    'packets': 'Total de pacotes',
    'src_ip_entropy': 'E(IP de origem)',
    'src_port_entropy': 'E(porta de origem)',
    'dst_ip_entropy': 'E(IP de destino)',
    'dst_port_entropy': 'E(porta de destino)'
    }


    titles = {
        'bits': 'Total de bits',
        'packets': 'Total de pacotes',
        'src_ip_entropy': 'Entropia IP de origem',
        'src_port_entropy': 'Entropia de porta de origem',
        'dst_ip_entropy': 'Entropia de IP de destino',
        'dst_port_entropy': 'Entropia de porta de destino'
    }

    attack_intervals = {
        '1': [],
        '2': [('10:15:00', '11:30:00')],
        '3': [('09:45:00', '11:10:00'), ('17:37:00', '18:55:00')]
    }


    plt.rcParams['axes.formatter.use_locale'] = True
    mpl.rcParams['lines.linewidth'] = 0.5
    figure, plots = plt.subplots(3, 2, figsize=(10.80,7.20))
    #plt.suptitle(graph_name, fontsize=14)
    real_data = scaler.inverse_transform(real_data)
    real_data = pd.DataFrame(real_data, columns = ['bits','dst_ip_entropy','dst_port_entropy','src_ip_entropy','src_port_entropy','packets'])

    predicted_data = scaler.inverse_transform(predicted_data)
    predicted_data = pd.DataFrame(predicted_data, columns = ['bits','dst_ip_entropy','dst_port_entropy','src_ip_entropy','src_port_entropy','packets'])


    locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
    import matplotlib.ticker as tkr
    def func(x, pos):  # formatter function takes tick label and tick position
        return locale.format_string("%.1f", x)
    axis_format = tkr.FuncFormatter(func)  # make formatter

    lin_space = np.linspace(0, 24, real_data.shape[0])
    for k in range(0, 6):
        i = k // 2 # mapping array index to matrix indices. (obs: 2 == matrix line length)
        j = k % 2
        plots[i][j].set_title(titles[features_names[k]])
        plots[i][j].set(xlabel="Tempo (hora)",ylabel=y_labels[features_names[k]])

        real = np.array(real_data[features_names[k]])
        plots[i][j].step(lin_space, real, label='Tráfego real', color='darkgreen')
        
        pred = np.array(predicted_data[features_names[k]])
        plots[i][j].step(lin_space, pred, label='Tráfego previsto', color='orange', alpha=0.9)


        plots[i][j].set_xticks(np.arange(0, 25, 2))
        plots[i][j].margins(x=0)

        #plots[i][j].legend(fontsize='xx-small')
        # mse = round(mean_squared_error(real_data[:, k], predicted_data[:, k]), 6)
        # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # plots[i][j].text(0.05, 0.90, f"MSE = {mse}", 
        #     transform=plots[i][j].transAxes, fontsize=7, bbox=props) 
        
        if day == '3':

            start, end = \
                    time_to_seconds(attack_intervals[day][0][0]), time_to_seconds(attack_intervals[day][0][1])
            portscan_duration = end-start

            start, end = \
                    time_to_seconds(attack_intervals[day][1][0]), time_to_seconds(attack_intervals[day][1][1])
            start = start - portscan_duration 
            end = end - portscan_duration
            
            start, end = lin_space[start], lin_space[end]
            plots[i][j].axvspan(start, end, label="Intervalo anômalo", color='r', alpha=0.25)
            #plots[i][j].step(lin_space[start:end], real[start:end], color='darkred')
            # plots[i][j].fill_between(
            #     lin_space[start:end], 
            #     real[start:end],
            #     -10,
            #     color= "r",
            #     alpha= 0.2)    
        else:
            for attack_interval in attack_intervals[day]:
                start, end = \
                    time_to_seconds(attack_interval[0]), time_to_seconds(attack_interval[1])

                #plots[i][j].step(lin_space[start:end], real[start:end], color='darkred')
                # plots[i][j].fill_between(
                # lin_space[start:end], 
                # real[start:end],
                # -10,
                # color= "r",
                # alpha= 0.2)  
                start, end = lin_space[start], lin_space[end]
                plots[i][j].axvspan(start, end, label="Intervalo anômalo", color='r', alpha=0.25)

        if k == 0:
            plots[i][j].set_ylim([0, 250000])
            plots[i][j].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        elif k == 1:
            plots[i][j].set_ylim([0, 1500])
            plots[i][j].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        else:
            plots[i][j].set_ylim([-0.5, 8.0])
            plots[i][j].yaxis.set_major_formatter(axis_format) #when using ','

    handles, labels = plots[2, 1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower left")
    plt.tight_layout(pad=1)
    figure.subplots_adjust(bottom=0.15)
    plt.savefig(save_path, format='jpg', dpi=800)
    plt.close()

