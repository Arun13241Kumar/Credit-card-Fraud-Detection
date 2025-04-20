import pickle
import numpy as np
import streamlit as st

#load model
loaded_model = pickle.load(open('model.pkl', 'rb'))

# evaluate model performance
#train_acc = accuracy_score(model.predict(X_train), y_train)
#test_acc = accuracy_score(model.predict(X_test), y_test)

#web app
st.title("Credit Card Fraud Detection Model")
input_df=st.text_input('Enter All required Feature Values')
input_df_splited = input_df.split(',')


submit = st.button("Submit")


if submit:
    features = np.array(input_df_splited,dtype=np.float64)
    prediction = loaded_model.predict(features.reshape(1,-1))
    
    if prediction[0] ==0:
        st.write("Legitimate Transaction")
    else:
        st.write("Fraudulent Transaction")