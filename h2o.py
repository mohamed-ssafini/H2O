# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 19:00:25 2019

@author: ssafini
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.options.display.width = 0
pd.set_option('display.expand_frame_repr', False)

import h2o
h2o.init()

#url="http://h2o-public-test-data.s3.amazonaws.com/smalldata/iris/iris_wheader.csv"
#iris=h2o.import_file(url)
wine_df = h2o.import_file("winequality-red.csv",sep = ";", destination_frame="wine_df")
from IPython.display import display
pd.set_option('display.expand_frame_repr', False)
wine_df
print(wine_df.head(5))# The default head() command displays the first 10 rows.
display(wine_df)
for col in wine_df.columns:
    wine_df[col].hist()
    

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,10))
corr = wine_df.cor().as_data_frame()
corr.index = wine_df.columns
sns.heatmap(corr, annot = True, cmap='RdYlGn', vmin=-1, vmax=1)
plt.title("Correlation Heatmap", fontsize=16)
plt.show()



train, test = wine_df.split_frame([0.8])
train.summary()
test.summary()
train.nrows
test.nrows

#train["quality"] = train["quality"].asfactor()
#test["quality"] = test["quality"].asfactor()

train.shape[0]
test.shape[0]
predictors = wine_df.columns[:-1]
response = "quality"
# RandomForest
from h2o.estimators.random_forest import H2ORandomForestEstimator
mRF=H2ORandomForestEstimator()
mRF.train(x=predictors, y=response, training_frame=train)
print(mRF)
mRF.model_performance(test)
#GBM
from h2o.estimators.gbm import H2OGradientBoostingEstimator
gbm = H2OGradientBoostingEstimator()
gbm.train(x=predictors, y=response, training_frame=train)
print(gbm)
gbm.model_performance(test) 
#GLM
# Import the fuction for GLM
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

# Set up GLM for regression
glm = H2OGeneralizedLinearEstimator(family = 'gaussian', model_id = 'glm_default')

# Use .train() to build the model
glm.train(x = predictors, 
                  y = 'quality', 
                  training_frame = wine_df)
print(glm)

glm.model_performance(test) 
#AutoML
from h2o.automl import H2OAutoML
aml = H2OAutoML(max_models = 20, max_runtime_secs=100, seed = 1)
aml.train(x=predictors, y=response, training_frame=train, validation_frame=test)
print(aml.leaderboard)
aml.leaderboard