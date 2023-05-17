'''
Author: Adam Forestier
Date: March 27, 2023
Notes
    - oop approach to creating matplotlib figures
    - plt.figure() creates a figure object
'''

import matplotlib.pyplot as plt
import numpy as np

# axis
a = np.linspace(0, 10, 11)
b = a**4
x = np.arange(0,10)
y = x * 2

# create Figure object
fig = plt.figure(figsize=(8, 6), dpi=150) # dpi is dots per inch

# Large axes
axes1 = fig.add_axes([0,0,1,1]) # 0, 0 is starting point. 0, 0 is the bottem left 1, 1 is the width and height of the canvas
axes1.set_xlim(0, 8)
axes1.set_ylim(0, 8000)
axes1.set_xlabel('A')
axes1.set_ylabel('B')
axes1.set_title('Power of 4')

# Small axes 
axes2 = fig.add_axes([0.2, 0.2, 0.5, 0.5])
axes2.set_xlim(1, 2)
axes2.set_ylim(0, 50)
axes2.set_xlabel('A')
axes2.set_ylabel('B')
axes2.set_title('Zoomed in')

# plot
axes1.plot(a, b)
axes2.plot(a, b)

# Export figure
fig.savefig('D:\\coding\\udemy\\python_datascience_machine_learning\\Matplotlib\\lessons\\images\\myfigureobjectexample.png', bbox_inches='tight')