# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 08:49:44 2022

@author: Phani Ullamgunta
"""

import pandas as pd
import streamlit as st 
from statsmodels.tsa.arima_model import ARIMA
from pickle import load
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.title('Internet Traffic Forecasting App')

st.sidebar.header('User Input')
def getNum(inp_str):
    num = ""
    for c in inp_str:
        if c.isdigit():
            num = num + c
    return num
 
def user_input_features():
    number = st.sidebar.number_input("Days to Forecast")
    data = number
    #features = pd.DataFrame(data,index = [0])
    return data

def main():
    df = user_input_features()
    
    if(st.sidebar.button(label='Forecast')):
        model = load(open('internet_traffic_Model.sav', 'rb'))
        forecast = model.forecast(steps = int(df))[0]
        st.subheader('Future Daily Visitors')
        foreval = pd.DataFrame(forecast, columns = ['forecasted Daily Visitors'])
        st.write(foreval)
        a = 173 + df

        st.subheader('plot for Future Daily Visitors')
        st.write(model.plot_predict(1,int(a)))
        
if __name__ == '__main__':
    main()
    
