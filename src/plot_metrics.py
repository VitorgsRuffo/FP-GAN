from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# #plotting GAN confusion matrix...
# cm = np.array([[4, 1],
#                [1, 2]])
# fig, ax = plot_confusion_matrix(conf_mat=cm, class_names=['Anômalo', 'Benigno'], figsize=(8, 8), cmap=plt.cm.Greens)
# plt.xlabel('Classificação inferida', fontsize=15)
# plt.ylabel('Classificação real', fontsize=15)
# plt.savefig(f"./gan_cm.png", dpi=800)
# plt.close()



# #plotting GRU confusion matrix...
# cm = np.array([[4652, 40],
#                [29, 76568]])


# fig, ax = plot_confusion_matrix(conf_mat=cm, class_names=['Anômalo', 'Benigno'], figsize=(8, 8), cmap=plt.cm.Greens)
# plt.xlabel('Classificação inferida', fontsize=15)
# plt.ylabel('Classificação real', fontsize=15)
# plt.savefig(f"./gru_cm.png", dpi=800)
# plt.close()




#plotting metrics histogram...
import numpy as np
import matplotlib.pyplot as plt
import locale
plt.rcParams['axes.formatter.use_locale'] = True

nids_gan = [0.9997, 0.9959, 0.9978, 0.9977]
nids_rnn = [0.9913, 0.9982, 0.9947, 0.9944] #gru
#nids_rnn = [0.9913, 0.9993, 0.9953, 0.9949] #lstm
  
n=4
r = np.arange(n)
width = 0.25


  
figure, plot = plt.subplots(1, 1, figsize=(10.80,7.20))

locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
import matplotlib.ticker as tkr
def func(x, pos):  # formatter function takes tick label and tick position
    return locale.format_string("%.1f", x)
axis_format = tkr.FuncFormatter(func)  # make formatter


plot.yaxis.set_major_formatter(axis_format) #when using ','


bars = plot.bar(r, nids_gan, color = '#3C6255',
        width = width, edgecolor = 'black',
        label='NIDS-GAN')

for b in bars:
        height = b.get_height()
        plot.text(x=b.get_x() + b.get_width() / 5, y=float(height)+0.01, s=f"{height:,}".replace(".", ","), ha='center', fontsize=15)
        



bars = plot.bar(r + width, nids_rnn, color = '#EAE7B1',
        width = width, edgecolor = 'black',
        label='NIDS-GRU')

for b in bars:
        height = b.get_height()
        plot.text(x=b.get_x() + b.get_width()*1.7 / 2, y=float(height)+0.01, s=f"{height:,}".replace(".", ","), ha='center', fontsize=15)



#plot.set(xlabel="Métrica",ylabel="Valor")
plt.xlabel("Métrica", fontsize=14)
plt.ylabel("Valor", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


#plot.title("Number of people voted in each year")
#plot.set_ylim([0.5, 1.0])  
plot.set_xticks(r + width/2,['Precision','Recall','F1-score','MCC'])

handles, labels = plot.get_legend_handles_labels()
figure.legend(handles, labels, loc="upper right", fontsize=15)

plt.savefig(f"./metrics.png", dpi=800)
plt.close()
