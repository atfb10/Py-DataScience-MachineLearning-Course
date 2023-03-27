'''
Author: Adam Forestier
Date: March 27, 2023
Notes
    - When using a text editor instead of jupyter notebook: use plt.show() to display the plot
    - plt.show() shows 1 plot at a time. to see the next one, simply close the first one
'''

import numpy as np
import matplotlib.pyplot as plt 

x = np.arange(0,10)
y = x * 2

# Plot using functional programming approach
plt.plot(x,y)
plt.title('x and y plot')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.xlim(0, 8) # lower and upper limit on x axis
plt.ylim(0, 16) # lower and upper limit on y axis
plt.savefig('D:\\coding\\udemy\\python_datascience_machine_learning\\Matplotlib\\lessons\\images\\myexamplelot.png') # Save figure. Can be any image file type
plt.show()