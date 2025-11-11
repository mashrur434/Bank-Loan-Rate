from flask import Flask, render_template, request
import pandas as pd
import pickle

import numpy as np


app = Flask(__name__)
data=pd.read_csv('Cleaned_data.csv')



# Load your trained model
model = pickle.load(open('RFModel.pkl', 'rb'))



@app.route('/')
def home():
     educations = sorted(data['person_education'].unique())
     homes=sorted(data['person_home_ownership'].unique())
     intents=sorted(data['loan_intent'].unique())
    
     return render_template('index.html',educations=educations,homes=homes,intents=intents)
    
    


       
@app.route('/predict', methods=['POST'])
def predict():
    try:
       Age = float(request.form.get('Age'))
       gender = request.form.get('gender')
       education = request.form.get('education')
       Income = float(request.form.get('Income'))
       experiance= float(request.form.get('experiance'))
       home = request.form.get('home')
       amount =float(request.form.get('amount'))
       intent =request.form.get('intent')
       interest = float(request.form.get('interest'))
       percent = float(request.form.get('percent'))
       Cred = float(request.form.get('Cred'))
       Credit = float(request.form.get('Credit'))
       Default=int(request.form.get('Default'))
       
       
       
       
       
       
       

       

       

       features = pd.DataFrame([[Age,
             gender,education, Income, experiance,home,amount,intent,
              interest,percent,Cred,Credit,Default
            ]], columns=[
            'person_age','person_gender', 'person_education', 'person_income', 'person_emp_exp',
             'person_home_ownership','loan_amnt','loan_intent','loan_int_rate',
             'loan_percent_income','cb_person_cred_hist_length','credit_score',
             'previous_loan_defaults_on_file'
              ])  

       
   

           

      

        # ✅ Do NOT encode manually
       
        # ✅ Let model pipeline handle encoding
       prediction = model.predict(features)[0]

        # Convert prediction to readable result
       if prediction == 0:
            result = "✅ Clear.Okay to Give loan"
      
       else:
            result =  "🚨 Fraud.Dont give Loan"

       return render_template('index.html', prediction=result)

    except Exception as e:
        raise e

       
       

















       


if __name__ == '__main__':
    app.run(debug=True)
