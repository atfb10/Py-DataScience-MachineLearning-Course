'''
Date: March 27, 2023
Notes
    - plt.subplots() is buiilt in method that will line up plots side by side 
    - plt.subplots() returns a tuple of fig (entire Figure object) and axes (numpy array holding each of the axes according to the position in the overall canvas)!
    - You just pass nrows and ncols
'''

import matplotlib.pyplot as plt
import numpy as np

a = np.linspace(0, 10, 11)
b = a**4
x = np.arange(0,10)
y = x * 2
w = x
z = w**2
r = a
q = a**3

# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=2)

# plot a 1x2
axes[0].plot(x, y)
axes[1].plot(a,b)
plt.tight_layout()

# plot a 2*2
fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (4,4), dpi=150)
axes[0][0].plot(x,y)
axes[0][1].plot(a,b)
axes[1][0].plot(w,z)
axes[1][1].plot(r,q)

# adjustments
axes[0][0].set_title('Title for top left')
axes[0][1].set_ylabel('B in top right')
axes[1][0].set_xlabel('w in bottom right')
axes[1][1].set_ylabel('Q in bottom right')
fig.suptitle('2 x 2 Subplot')

fig.subplots_adjust(wspace=0.5, hspace=0.5)
fig.savefig('D:\\coding\\udemy\\python_datascience_machine_learning\\Matplotlib\\lessons\\images\\suplotexample.png', bbox_inches='tight')
plt.show()