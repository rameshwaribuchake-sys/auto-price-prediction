import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE=Path(__file__).resolve().parent
dt=joblib.load(BASE/'dt_reg.pkl')
rf=joblib.load(BASE/'rf_model.pkl')
features=joblib.load(BASE/'feature_columns.pkl')
info=joblib.load(BASE/'preprocess_info.pkl')

st.set_page_config(page_title='Auto Price Prediction Dashboard', page_icon='🚗', layout='wide')

st.title('🚗 Auto Price Prediction Dashboard')
st.caption('Decision Tree and Random Forest Regression | Based on the Auto dataset and notebook preprocessing')

# Metrics from the same train/test split used for the notebook
X_all=pd.read_csv(BASE/'processed_auto.csv')[features]
y_all=pd.read_csv(BASE/'processed_auto.csv')['price']
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
X_train,X_test,y_train,y_test=train_test_split(X_all,y_all,test_size=0.2,random_state=42)

def scores(model):
    p=model.predict(X_test)
    return r2_score(y_test,p), np.sqrt(mean_squared_error(y_test,p)), mean_absolute_error(y_test,p)

dt_r2,dt_rmse,dt_mae=scores(dt)
rf_r2,rf_rmse,rf_mae=scores(rf)

c1,c2,c3,c4=st.columns(4)
c1.metric('Random Forest R²',f'{rf_r2:.3f}')
c2.metric('RF RMSE',f'${rf_rmse:,.0f}')
c3.metric('Decision Tree R²',f'{dt_r2:.3f}')
c4.metric('Dataset Rows',f'{len(X_all)}')

st.sidebar.header('Prediction Settings')
model_name=st.sidebar.selectbox('Select Model',['Random Forest','Decision Tree'])
model=rf if model_name=='Random Forest' else dt

st.sidebar.subheader('Vehicle Features')

def num(label, default, minv, maxv, step):
    return st.sidebar.number_input(label, min_value=float(minv), max_value=float(maxv), value=float(default), step=float(step))

symboling=num('Symboling',0,-3,3,1)
wheel_base=num('Wheel Base',98.0,86,121,0.1)
length=num('Length',174.0,141,209,0.1)
width=num('Width',65.0,60,72,0.1)
height=num('Height',54.0,47,60,0.1)
curb_weight=num('Curb Weight',2550,1400,4100,10)
cylinders=st.sidebar.selectbox('Number of Cylinders',[2,3,4,5,6,8,12],index=2)
engine_size=num('Engine Size',120,60,330,1)
compression=num('Compression Ratio',10.0,7,24,0.1)
horsepower=num('Horsepower',104,48,300,1)
peak_rpm=num('Peak RPM',5125,4000,7000,50)
city_mpg=num('City MPG',25,13,50,1)
highway_mpg=num('Highway MPG',30,16,55,1)

row=pd.DataFrame([[symboling,wheel_base,length,width,height,curb_weight,cylinders,engine_size,compression,horsepower,peak_rpm,city_mpg,highway_mpg]],columns=features)

st.subheader('Price Prediction')
if st.button('🔮 Predict Car Price', type='primary', use_container_width=True):
    pred=float(model.predict(row)[0])
    st.success(f'Estimated Car Price: **${pred:,.2f}**')
    st.dataframe(row, use_container_width=True, hide_index=True)

st.subheader('Model Comparison')
comparison=pd.DataFrame({
    'Model':['Decision Tree','Random Forest'],
    'R² Score':[dt_r2,rf_r2],
    'RMSE':[dt_rmse,rf_rmse],
    'MAE':[dt_mae,rf_mae]
})
st.dataframe(comparison.style.format({'R² Score':'{:.3f}','RMSE':'${:,.0f}','MAE':'${:,.0f}'}),use_container_width=True,hide_index=True)

st.subheader('Feature Importance')
imp=pd.DataFrame({'Feature':features,'Importance':model.feature_importances_}).sort_values('Importance',ascending=False)
st.bar_chart(imp.set_index('Feature'))

st.info('Note: The preprocessing and 13 input features follow the uploaded notebook. Missing horsepower/peak-rpm values were filled using training-dataset column means.')
