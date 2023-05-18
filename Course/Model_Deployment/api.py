import joblib
import pandas as pd

from flask import Flask, jsonify, request

# Create Flask APP
app = Flask(__name__)

# CONNECT POST API CALL --> predict() Function
@app.route('/predict', methods=['POST'])
def predict():
    # GET JSON REQUEST
    feature_data = request.json

    # CONVERT JSON to PANDAS DF (column names)
    df = pd.DataFrame(feature_data)
    df = df.reindex(columns=col_names)

    # PREDICT
    prediction = list(model.predict(df))

    # return PREDICTION AS JSON
    return jsonify({'preidction': str(prediction)})


# LOAD MY MODEL and LOAD COLUMN NAMES
if __name__ == '__main__':
    model = joblib.load('final_model.pkl')
    col_names = joblib.load('col_names.pkl')

    app.run(debug=True)

'''
NOTE - steps flask goes through
1. Imports
2. Create app
3. run app
4. Handle all routing functions
''' 