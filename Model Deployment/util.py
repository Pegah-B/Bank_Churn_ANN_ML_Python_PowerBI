import pickle
import json
import numpy as np
import pandas as pd

def predict_churn(params):

    x = np.zeros((1,(len(data_columns))))

    x[0,0] = params.get('credit_score') 
    x[0,1] = params.get('age') 
    x[0,2] = params.get('tenure') 
    x[0,3] = params.get('balance') 
    x[0,4] = params.get('products_number') 
    x[0,5] = params.get('credit_card') 
    x[0,6] = params.get('active_member') 
    x[0,7] = params.get('estimated_salary') 

    country_col = 'country_' + params.get('country')
    x[0,data_columns.index(country_col)] = 1

    gender_col = 'gender_' + params.get('gender')
    x[0,data_columns.index(gender_col)] = 1

    x_df = pd.DataFrame(x, columns=data_columns)

    y_pred = model.predict(x_df)
    y_prob = model.predict_proba(x_df)[:,1]

    return (y_pred, y_prob)


def load_artifacts(model_version):
    print('loading saved artifacts')
    global data_columns
    global model

    model_path = f'../Model Training/model_v{model_version}.pickle'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open('../Model Training/data_columns.json', 'r') as f:
        data_columns = json.load(f)['data_columns']    


if __name__ == '__main__':  
    model_version = 1  
    load_artifacts(model_version)  
    params = {
    'age': 45,
    'gender': 'Male',
    'country': 'France',
    'credit_card' : 1,
    'active_member': 1,
    'products_number': 2,
    'tenure': 4,
    'credit_score': 720,
    'balance': 12000,
    'estimated_salary': 75000
    }
    y_pred, y_prob = predict_churn(params)
    print("Prediction: no churn" if y_pred[0] == 0 else "Prediction: churn")
    print(f'Churn Probability: {y_prob[0] * 100 :0.2f}%')

    
