#pip install streamlit  

import pickle
import streamlit as st
import numpy as np

# Load the saved trained ML models
lr_reg = pickle.load(open('lr_model.pkl','rb')) 
knr_reg = pickle.load(open('knr_model.pkl','rb'))
decision_tree = pickle.load(open('dtr_model.pkl','rb')) 
random_forest = pickle.load(open('rf_model.pkl','rb'))

# rb = 'read binary'
st.title('ML Web App - Car Price Prediction')

ml_model = ['Linear Regression','KNeighbors Regressor','DecisionTree Regressor','RandomForest Regressor']
option = st.sidebar.selectbox('Select the ML model which you want to use', ml_model)

# ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']

selling_price = st.slider('Select Selling price', 200000, 1000000, step = 100000)
km_driven = st.slider('Select km driven',100, 1500000, step = 100000)
fuel = st.slider('Select fuel type',1, 7, step = 1)
seller_type= st.slider('Select seller type',0, 2, step = 1)
owner = st.slider('select owner type'.0, 3,step = 1)
test  = [[selling_price, km_driven, fuel, seller_type, owner]]
st.write('Test_Data', test)
st.write('Option', option)

if st.button('Predict'):
    if option=="Linear Regression":
        st.success('Label is :' + lr_reg.predict(test)[0],  icon="✅")
    elif option=="DecisionTree Regressor":
        st.success(decision_tree.predict(test)[0])
    else:
        st.success(random_forest.predict(test)[0])



# Terminal commands
# To run Streamlit web App - streamlit run app.py
# To stop the Server - Ctrl + C



