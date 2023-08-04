from explainerdashboard import ExplainerDashboard, ClassifierExplainer, RegressionExplainer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from explainerdashboard.custom import *
from sklearn.model_selection import RepeatedKFold
from dash_bootstrap_components.themes import FLATLY

import dash_bootstrap_components as dbc



#Importing Libraries & Packages


import openpyxl

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from pandas import DataFrame

#import portalocker
from dash import Dash, callback, html, dcc, dash_table, Input, Output, State, MATCH, ALL



#Import data
#############################################################################################
##### use the below data for both core and headline inflation

################### for Prices-more exogenous variable ################

#data = pd.read_excel('Data.xlsx', sheet_name="year").dropna()


#X = data.drop(['Headline Inflation', 'Core_farm', 'Core_farm_energy', 'Date'], axis=1)


######## switch y between core and headline

#y = data['Core_farm']

#y = data['Headline Inflation']
#############################################################################################


################### for others without extral CPI ################

data = pd.read_excel('Data.xlsx', sheet_name="2001-nocpi").dropna()

X = data.drop(['Headline Inflation', 'Date'], axis=1)

y = data['Headline Inflation']



##################################################
##### For Core Inflation 
######################################################
#X = data.drop(['Headline Inflation', 'Food', 'Core_farm', 'Core_farm_energy','Date'], axis=1)

#y = data['Core_farm']


#Food.DataFrame(data.target,columns=["target"])


########## Dataset Split ########


X_train = X[X.index < 90]
y_train = y[y.index < 90]              
    
X_test = X[X.index >= 90]    
y_test = y[y.index >= 90]


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)



model = RandomForestRegressor(n_estimators = 400,
                           n_jobs = -1,
                           oob_score = True,
                           bootstrap = True,
                            max_depth=5,
                           random_state = 42)



model.fit(X_train, y_train.values.ravel())


model.score (X_train,y_train), model.score(X_test,y_test),model.oob_score_


#X_train, y_train, X_test, y_test = titanic_survive()
#train_names, test_names = titanic_names()
#model = RandomForestClassifier(n_estimators=50, max_depth=5)
#model.fit(X_train, y_train)




explainer = RegressionExplainer(model, X, y)


ExplainerDashboard(explainer).run(2000)


