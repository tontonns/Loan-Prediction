import streamlit as st
import joblib
import numpy as np
import pandas as pd

XGBC = joblib.load('XGBCLoan.pkl')
EncGen= joblib.load('encode_gender.pkl')
EncPL=joblib.load('encode_pl.pkl')
EncLI=joblib.load('li_enc.pkl')
EncPE=joblib.load('pe_enc.pkl')
EncPH=joblib.load('ph_enc.pkl')

def main():
    st.title('Loan Model Deployment')

    person_age=st.number_input("Age", 0, 100)
    person_gender=st.radio("Gender", ["Male","Female"])
    person_education=st.radio("Level of Education", ["High School","Associate","Bachelor","Master","Doctorate"])
    person_income=st.number_input("Yearly Income", 0,1000000000)
    person_emp_exp=st.number_input("Years of Work Experience", 0,100)
    person_home_ownership=st.radio("Home Ownership Status", ["OWN","MORTGAGE","RENT","OTHER"])
    loan_amnt=st.number_input("Loan Amount Requested", 0,1000000000)
    loan_intent=st.radio("Loan Intention", ["EDUCATION","MEDICAL","VENTURE","PERSONAL","DEBTCONSOLIDATION","HOMEIMPROVEMENT"])
    loan_int_rate=st.number_input("Loan Interest Rate", 0,100)
    loan_percent_income=st.number_input("Loan Amount as a Percentage of Annual Income", 0.00,100.00)
    cb_person_cred_hist_length=st.number_input("Loan Duration in Years", 0,100)
    credit_score=st.number_input("Credit Score", 300,850)
    previous_loan_defaults_on_file=st.radio("Indicator of Previous Loan Delinquencies", ["Yes","No"])
    
    data = {'age' : int(person_age), 'gender' : person_gender, 'person_education' : person_education,
        'income' : int(person_income), 'experience' : person_emp_exp, 'person_home_ownership' : person_home_ownership,
        'loan_amnt' : int(loan_amnt), 'loan_intent' : loan_intent, 'loan_int_rate' : int(loan_int_rate),
        'loan_percent_income' : float(loan_percent_income), 'cb_person_cred_hist_length' : int(cb_person_cred_hist_length),
        'credit_score' : int(credit_score), 'previous_loan_defaults_on_file' : previous_loan_defaults_on_file}

    
    df=pd.DataFrame([list(data.values())], columns=['age','gender', 'person_education', 'income','experience', 
                                                'person_home_ownership', 'loan_amnt','loan_intent', 
                                                'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
                                                'credit_score', 'previous_loan_defaults_on_file'])

    df=df.replace(EncGen)
    df=df.replace(EncPL)
    cat_li=df[['loan_intent']]
    cat_pe=df[['person_education']]
    cat_ph=df[['person_home_ownership']]
    cat_enc_li=pd.DataFrame(EncLI.transform(cat_li).toarray(),columns=EncLI.get_feature_names_out())
    cat_enc_pe=pd.DataFrame(EncPE.transform(cat_pe).toarray(),columns=EncPE.get_feature_names_out())
    cat_enc_ph=pd.DataFrame(EncPH.transform(cat_ph).toarray(),columns=EncPH.get_feature_names_out())
    df=pd.concat([df,cat_enc_li,cat_enc_pe,cat_enc_ph], axis=1)
    df=df.drop(['loan_intent', 'person_education', 'person_home_ownership'],axis=1)
    
    if st.button('Make Prediction'):
        features=df      
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = XGBC.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()