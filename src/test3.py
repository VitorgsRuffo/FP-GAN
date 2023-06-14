import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


# Some example data to display
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

# Plot
fig = plt.figure(figsize=(14,5))
ax = plt.axes()
ax.plot(x, y)

# Label the axis
ax.set_xlabel('')
ax.set_ylabel('EUR/CHF')

#I want to select the x-range for the zoomed region. I have figured it out suitable values
# by trial and error. How can I pass more elegantly the dates as something like
x1 = 1
x2 = 2

# select y-range for zoomed region
y1 = 0.75
y2 = 1.0

# Make the zoom-in plot:
axins = zoomed_inset_axes(ax, 2, loc=4) # zoom = 2
axins.plot(x, y)
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
plt.xticks(visible=False)
plt.yticks(visible=False)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
plt.draw()
plt.show()
