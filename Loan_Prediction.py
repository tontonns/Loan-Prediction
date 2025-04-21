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
    person_home_ownership=st.radio("Home Ownership Status", ["Own","Mortgage","Rent","Other"])
    loan_amnt=st.number_input("Loan Amount Requested", 0,1000000000)
    loan_intent=st.radio("Loan Intention", ["Education","Medical","Venture","Personal","DebtConsolidation","HomeImprovement"])
    loan_int_rate=st.number_input("Loan Interest Rate", 0,100)
    loan_percent_income=st.number_input("Loan Amount as a Percentage of Annual Income", 0,100)
    cb_person_cred_hist_length=st.number_input("Loan Duration in Years", 0,100)
    credit_score=st.number_input("Credit Score", 300,850)
    previous_loan_defaults_on_file=st.radio("Indicator of Previous Loan Delinquencies", ["Yes","No"])
    
    data = {'Age' : int(person_age), 'Gender' : person_gender, 'Education' : person_education,
            'Income' : int(person_income), 'Experience' : person_emp_exp, 'Home Ownership' : person_home_ownership,
            'Loan Amount' : int(loan_amnt), 'Loan Intention' : loan_intent, 'Loan Interest Rate' : int(loan_int_rate),
            'Loan Percent Income' : float(loan_percent_income), 'Credit Duration' : int(cb_person_cred_hist_length),
            'Credit Score' : int(credit_score), 'Previous Loan' : previous_loan_defaults_on_file}
    
    df=pd.DataFrame([list(data.values())], columns=['Age','Gender', 'Education', 'Income','Experience', 
                                                'Home Ownership', 'Loan Amount','Loan Intention', 
                                                'Loan Interest Rate', 'Loan Percent Income', 'Credit Duration',
                                                'Credit Score', 'Previous Loan'])

    df=df.replace(EncGen)
    df=df.replace(EncPL)
    cat_li=df[['Loan Intention']]
    cat_pe=df[['Education']]
    cat_ph=df[['Home Ownership']]
    cat_enc_li=pd.DataFrame(EncLI.transform(cat_li).toarray(),columns=EncLI.get_feature_names_out())
    cat_enc_pe=pd.DataFrame(EncPE.transform(cat_pe).toarray(),columns=EncPE.get_feature_names_out())
    cat_enc_ph=pd.DataFrame(EncPH.transform(cat_ph).toarray(),columns=EncPH.get_feature_names_out())
    df=pd.concat([df,cat_enc_li,cat_enc_pe,cat_enc_ph], axis=1)
    df=df.drop(['Loan Intention', 'Education', 'Home Ownership'],axis=1)
    
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