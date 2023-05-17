# TASK: Use pandas to grab the expenses paid by Bob.
# MAKE SURE TO READ THE FULL INSTRUCTIONS ABOVE CAREFULLY, AS THE EVALUATION SCRIPT IS VERY STRICT.
#  Link to Solution: https://gist.github.com/Pierian-Data/3d7f7cb3528f015d9584d04a7168b97f
import pandas as pd
expenses = pd.Series({'Andrew':200,'Bob':150,'Claire':450})

# bob_expense = # Use pandas, don't just manually write in the number here.
bob_expense = expenses['Bob']