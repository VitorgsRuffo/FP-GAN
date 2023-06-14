#ROC curve:
import matplotlib as mpl
from pickle import load
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
mpl.rcParams['lines.linewidth'] = 1.2


_file = open('gan_normalized_raw_predictions.pkl', 'rb')
gan_pred = load(_file)
gan_pred = gan_pred[10:]

_file = open('gru_normalized_raw_predictions.pkl', 'rb')
gru_pred = load(_file)

_file = open('labels.pkl', 'rb')
labels = load(_file)



import locale
plt.rcParams['axes.formatter.use_locale'] = True


fig = plt.figure()
ax = plt.axes()


#gan
fpr1, tpr1, _ = metrics.roc_curve(labels,  gan_pred)
auc = metrics.roc_auc_score(labels, gan_pred)
ax.plot(fpr1,tpr1,label="NIDS-GAN\n(AUROC="+str(round(auc, 4)).replace('.', ',')+")\n", color="black")

#gru
fpr2, tpr2, _ = metrics.roc_curve(labels,  gru_pred)
auc = metrics.roc_auc_score(labels, gru_pred)
ax.plot(fpr2,tpr2,label="NIDS-GRU\n(AUROC="+str(round(auc, 4)).replace('.', ',')+")", color="#FC4F00")


locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
import matplotlib.ticker as tkr
def func(x, pos):  # formatter function takes tick label and tick position
    return locale.format_string("%.2f", x)
axis_format = tkr.FuncFormatter(func)  # make formatter

ax.yaxis.set_major_formatter(axis_format) #when using ','
ax.set_ylabel('True Positive Rate (Recall)')
ax.set_xlabel('False Positive Rate (FPR)')
ax.legend(loc=4, fontsize=10)


#I want to select the x-range for the zoomed region. I have figured it out suitable values
# by trial and error. How can I pass more elegantly the dates as something like
x1 = -0.02
x2 = 0.03

# select y-range for zoomed region
y1 = 0.97
y2 = 1.02

# Make the zoom-in plot:
axins = zoomed_inset_axes(ax, 8, loc=10) # zoom = 2
axins.plot(fpr1,tpr1, color="black")
axins.plot(fpr2,tpr2, color="#FC4F00")
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
plt.xticks(visible=False)
plt.yticks(visible=False)
mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="0.5")


#saving...
plt.draw()
plt.savefig("./roc.jpg", format='jpg', dpi=800)
plt.close()
