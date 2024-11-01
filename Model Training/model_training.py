
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras
import pickle

df = pd.read_csv('../Dataset and Notebooks/Bank_Churn_Dataset.csv')
df.drop('customer_id' , axis=1 , inplace=True)

#one-hot encoding
df2 = pd.get_dummies(data=df, columns=['country','gender'] , dtype=int) 

#Feature Scaling
col_to_scale = ['credit_score','tenure','balance','products_number','age','estimated_salary']
scaler = MinMaxScaler()
df2[col_to_scale] = scaler.fit_transform(df2[col_to_scale])

#Features and Target variable
x = df2.drop('churn',axis=1)
y = df2['churn']

#Save Columns Names as JSON
data_columns = [col for col in x.columns]

import json
columns = {'data_columns' : data_columns}
with open ('data_columns.json' , 'w') as f:
    json.dump(columns , f)

#------------------------------------------
#Train ML Models
#------------------------------------------
#Split the Dataset for Training and Testing 
x_train , x_test , y_train , y_test = train_test_split(x , y, train_size = 0.8 , random_state = 42)

#Apply oversampling to the minority class using SMOTE
smote = SMOTE(sampling_strategy='minority')
x_sm, y_sm = smote.fit_resample(x_train, y_train)

ml_model = {
    'RF'  : RandomForestClassifier(),
    'XGB' : XGBClassifier()
}
# Hyperparameters selected based on GridSearch done in the notebook file "Bank_Churn_Predict_ML.ipynb"
ml_hyper_params = {
    'RF': {
        'bootstrap': True,
        'ccp_alpha': 0.0,
        'class_weight': None,
        'criterion': 'gini', 
        'max_depth': None, 
        'max_features': 'sqrt',
        'max_leaf_nodes': None, 
        'max_samples': None, 
        'min_impurity_decrease': 0.0,
        'min_samples_leaf': 1, 
        'min_samples_split': 2, 
        'min_weight_fraction_leaf': 0.0, 
        'n_estimators': 100, 
        'n_jobs': None, 
        'oob_score': False, 
        'random_state': None, 
        'verbose': 0, 
        'warm_start': False
    },
    'XGB': {
        'objective': 'binary:logistic',
        'base_score': None,
        'booster': None, 
        'callbacks': None, 
        'colsample_bylevel': None, 
        'colsample_bynode': None, 
        'colsample_bytree': None, 
        'device': None, 
        'early_stopping_rounds': None, 
        'enable_categorical': False, 
        'eval_metric': None, 
        'feature_types': None, 
        'gamma': 0.01, 
        'grow_policy': None, 
        'importance_type': None, 
        'interaction_constraints': None, 
        'learning_rate': 0.1, 
        'max_bin': None, 
        'max_cat_threshold': None, 
        'max_cat_to_onehot': None, 
        'max_delta_step': None, 
        'max_depth': 7, 
        'max_leaves': None, 
        'min_child_weight': None, 
        'missing': np.nan, 
        'monotone_constraints': None, 
        'multi_strategy': None, 
        'n_estimators': 200, 
        'n_jobs': None, 
        'num_parallel_tree': None, 
        'random_state': None, 
        'reg_alpha': None, 
        'reg_lambda': None, 
        'sampling_method': None, 
        'scale_pos_weight': None, 
        'subsample': None, 
        'tree_method': None, 
        'validate_parameters': None, 
        'verbosity': None
    }
}

#Train and Save ML Models
model_filenames = {
    'RF': 'model_v1.pickle',
    'XGB': 'model_v2.pickle'
}
for model_name, model in ml_model.items():
    print(f"Training started for {model_name} model...")
    model.set_params(**ml_hyper_params[model_name])
    model.fit(x_train, y_train)

    with open(model_filenames[model_name], 'wb') as file:
        pickle.dump(model, file)

    print(f"{model_name} model trained and saved as {model_filenames[model_name]}.")

#------------------------------------------
#Train ANN Model
#------------------------------------------
#Split the Dataset for Training and Testing 
x_train, x_rem, y_train, y_rem = train_test_split(x,y, test_size=0.4, random_state=42, stratify=y)
x_val, x_test, y_val, y_test = train_test_split(x_rem,y_rem, test_size=0.5, random_state=42, stratify=y_rem)

#Apply oversampling to the minority class using SMOTE
smote = SMOTE(sampling_strategy='minority')
x_sm, y_sm = smote.fit_resample(x_train, y_train)    

model = keras.Sequential([
    keras.layers.Input(shape=(13,)), 
    keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid')  
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print(f"Training started for ANN model...")
model.fit(x = x_sm, y = y_sm,
          validation_data = (x_val,y_val),
          epochs=100,
          callbacks=[early_stopping]) 

#Save ANN Model
with open('model_v3.pickle' , 'wb') as f:
    pickle.dump(model , f)

print(f"ANN model trained and saved as model_v3.pickle.")    