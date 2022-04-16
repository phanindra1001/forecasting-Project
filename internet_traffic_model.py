# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 08:43:53 2022

@author: Phani Ullamgunta
"""

import pandas as pd
import streamlit as st 
from statsmodels.tsa.arima_model import ARIMA
from pickle import dump
import warnings
warnings.filterwarnings('ignore')

data1 = pd.read_csv('P:/Project/internet traffic prediction/cleaned_data.csv')

model = ARIMA(data1['Daily_Visitors'],order=(7,1,5)).fit()

dump(model, open('internet_traffic_Model.sav', 'wb'))
