from flask import Flask, request, jsonify, render_template
import util

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_churn', methods = ['POST'])
def predict_churn():
    
    params = {
        'age': int(request.form['age']),
        'gender': request.form['gender'],
        'country': request.form['country'],
        'credit_card': int(request.form['credit_card']),
        'active_member': int(request.form['active_member']),
        'products_number': int(request.form['products_number']),
        'tenure': int(request.form['tenure']),
        'credit_score': int(request.form['credit_score']),
        'balance': float(request.form['balance']),
        'estimated_salary': float(request.form['estimated_salary'])  
    }

    y_pred, y_prob = util.predict_churn(params)
    prediction = "Prediction: No Churn" if y_pred[0] == 0 else "Prediction: Churn"
    churn_prob = f'Churn Probability: {y_prob[0] * 100:.2f}%'    

    return render_template('index.html', params=params, prediction=prediction, churn_prob=churn_prob)

if __name__ == '__main__' : 
    print("Starting Python Flask Server on port 8000...")
    model_version = 1
    util.load_artifacts(model_version)
    app.run(port = 8000, debug=True)

    
