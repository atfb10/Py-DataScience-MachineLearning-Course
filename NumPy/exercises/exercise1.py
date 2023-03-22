# TASK: Create a numpy array called myarray which consists of 101 evenly linearly spaced points between 0 and 10.
# MAKE SURE TO READ THE FULL INSTRUCTIONS ABOVE CAREFULLY, AS THE EVALUATION SCRIPT IS VERY STRICT.
# Link to Solution: https://gist.github.com/Pierian-Data/ea9c4d2fc6c98ac74af18134cd924867
# import ?
# myarray = ?

import numpy as np

myarray = np.linspace(0, 10, 101)
for i in myarray: print(i)