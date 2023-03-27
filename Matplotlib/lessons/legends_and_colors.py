'''
Author: Adam Forestier
Date: March 27, 2023
Notes:
'''
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(4, 4))
axes = fig.add_axes([0,0,1,1])

x = np.linspace(0,10,11)

axes.set_xlabel('x')
axes.set_title('X by x^2 by x^3')

# label
axes.plot(x, x, label='x vs x', color='green', linewidth=1, marker='o', markerfacecolor='red', markeredgewidth=19, markeredgecolor='orange') # color via text
axes.plot(x, x**2, label='x vs x squared', color='#650dbd', lw=2, linestyle='-', marker='v', ms=5) # color w/ RGB Hex code. lw=linewidth. ms = markersize
axes.plot(x, x**3, label='x vs x cubed', color='#010dbd', lw=5, ls='-.', marker='h', markersize=10) # color w/ RGB Hex code. ls = linestyle

plt.show()