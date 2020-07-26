#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sqlite3
import numpy as np


# # 1. Import, Restructure, and Clean Data

# In[2]:


#define connection
conn = sqlite3.connect('database.sqlite')


# In[3]:


#write query for first import (weather)
query = """SELECT date, mean_temperature_f, precipitation_inches, mean_wind_speed_mph FROM weather"""


# In[4]:


#read in the weather table. set index as date to prevent multiindex default
weather = pd.read_sql_query(query, conn, parse_dates = ['date'], index_col = 'date')
weather.reset_index(inplace = True)
weather.replace("", np.nan)


# In[5]:


#drop NAs from the weather table
weather.dropna(inplace = True)


# In[6]:


#sort weather df by date, and drop the duplicates, keeping the first instance of each
weather.sort_values(by = 'date', inplace = True)
weather = weather[weather.mean_temperature_f != '']
weather_drop = weather.drop_duplicates(subset = 'date', keep = 'first').reset_index().drop('index', axis=1)


# In[7]:


# replace all instance of T (trace) in the precip column with 0.005
weather_drop['precipitation_inches'].replace('T', 0.005, inplace = True)


# In[8]:


#convert precip and temp/wind cols to flt and int dtypes respectively
weather_drop['precipitation_inches'].astype('float')
weather_drop[['mean_temperature_f', 'mean_wind_speed_mph']] = weather_drop[['mean_temperature_f', 'mean_wind_speed_mph']].astype('int')


# In[9]:


#extract year and day number from date col, and drop date col
weather_drop['dayofyear'] = weather_drop['date'].dt.dayofyear.astype(int)
weather_drop['year'] = weather_drop['date'].dt.year.astype(int)
weather_drop.drop('date', axis = 1, inplace = True)
weather_drop.info()


# In[10]:


#read in station table from sql db
query = """SELECT id, lat, long, dock_count
            FROM station"""

stations = pd.read_sql_query(query, conn)
stations['dock_count'] = stations['dock_count'].astype(int)
stations.columns = ['station_id', 'lat', 'long', 'dock_count']
stations['station_id'] = stations['station_id'].astype(int)
stations.info()


# In[11]:


#read in status table in chunks of 10**6
status_chunk = pd.read_csv('status.csv', 
                           usecols = ['station_id', 'bikes_available', 'time'], 
                           parse_dates = ['time'], 
                           chunksize = 10**6,
                          iterator = True)

#create an empty df with the appropriate col names
status = pd.DataFrame(columns=['station_id', 'bikes_available', 'time'])

#use for loop to take samples of the chunks and append them to the the status df
for chunk in status_chunk:
    chunk = chunk.sample(frac = 0.1)
    status = status.append(chunk)


# In[12]:


#convert station_id and bikes_available columns to int type
status[['station_id', 'bikes_available']] = status[['station_id', 'bikes_available']].astype(int)


# In[13]:


status.info()


# In[14]:


#break down the datetime into its individual features and drop the original datetime
status['dayofweek'] = status['time'].dt.dayofweek
status['min_in_day'] = status['time'].dt.hour * 60 + status['time'].dt.minute
status['dayofyear'] = status['time'].dt.dayofyear
status['year'] = status['time'].dt.year

status.drop('time', axis = 1, inplace = True)

status = status.astype(int)

status.info()


# In[15]:


#merge status and weather dfs on 'dayofyear' and 'year'. 
status = status.merge(weather_drop, how = 'left', on = ['dayofyear', 'year'])

#merge status and stations dfs on station_id
status = status.merge(stations, how = 'left', on = 'station_id')


# In[16]:


#write the full merged dataframe to a csv
#status.to_csv('status_stn_wx_full.csv')


# In[17]:


status.info()


# In[ ]:





# # 3. Exploratory Data Analysis

# In[18]:


import matplotlib.pyplot as plt


# In[19]:


#take a randome sample of 1e6 data to get an idea of trends that are present
#status_sample = status.sample(10**6)
#status_sample.info()


# In[20]:


#plot differences in bike availability by day of the week
plt.plot(status.groupby("dayofweek").bikes_available.mean())


# In[21]:


#plot the histogram of bike availabilities at all stations
plt.hist(status["bikes_available"], bins = (status["bikes_available"].max() + 1))
plt.axvline(status["bikes_available"].median(), color = 'red')
plt.axvline(status["bikes_available"].median() + status["bikes_available"].std(), color = 'yellow')
plt.axvline(status["bikes_available"].median() - status["bikes_available"].std(), color = 'yellow')


# In[22]:


#plot how biek avavilability changes over the course of the day
plt.plot(status.groupby(["min_in_day"])["bikes_available"].mean())


# In[23]:


##plot differences in bike availability by day of the year
plt.plot(status.groupby("dayofyear").bikes_available.mean())


# # 4. Data Preprocessing

# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[25]:


#take a sample of size sample_size from status_merge to do test runs on ML model
#using the same var name so that this cell can be easily removed for a full run
sample_size = 1250000
status = status.sample(n = sample_size, random_state = 42)


# In[26]:


#instantiate standard scaler
scaler = StandardScaler()


# In[27]:


#scale min_in_day, lat, long, and dock_count cols
status[['mean_temperature_f','precipitation_inches', 'mean_wind_speed_mph', 'lat', 'long', 'dock_count', 'min_in_day', 'dayofyear']] = scaler.fit_transform(status[['mean_temperature_f','precipitation_inches', 'mean_wind_speed_mph', 'lat', 'long', 'dock_count', 'min_in_day', 'dayofyear']])


# In[28]:


#convert station_id and year to categeoricals, then one-hot encode them
status[['station_id', 'year', 'dayofweek']] = status[['station_id', 'year', 'dayofweek']] .astype('category')
status = pd.get_dummies(status)


# In[29]:


#create X and y
bikes = status['bikes_available']
status = status.drop('bikes_available', axis = 1)


# In[30]:


bikes = bikes.values
status = status.values


# In[31]:


#find the shape of X_scale to use as the input dimension, asigning the length and width to l and w
l, w = status.shape


# In[32]:


#create a train/test split of the data set being used in the model
X_train, X_test, y_train, y_test = train_test_split(status, bikes, test_size = 0.2, random_state = 21)


# # 5. Create Neural Net ML Model

# In[ ]:


#import necessary elements of ML model
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV


# In[34]:


#define a function that creates a keras model with parameters optimizer, 
#activation, nl (number of layers), and nn (number of neurons)
def create_model(opt = 'adam', act = 'relu', nl = 1, nn = 32):
    model = Sequential()
    model.add(Dense(nn, input_shape = (w,), activation = act))
    for i in range(nl):
        model.add(Dense(nn, activation = act))
    model.add(Dense(1, activation = act))
    model.compile(optimizer = opt, loss = 'mse', metrics = ['mse', 'mean_absolute_error'])
    return model
    
#create model as an sklearn regressor
model = KerasRegressor(build_fn = create_model, epochs = 5)

#set params to search over
params = dict(opt = ['adam'],
              act = ['relu'],
              nl = [1, 2, 3, 4],
              nn = [32, 64, 128])


# In[35]:


#create random search cv object
random_search = RandomizedSearchCV(model, param_distributions = params, cv = 5, n_iter = 10)

random_search_results = random_search.fit(status, bikes)


# In[36]:


#print the best score obtained from the RandomizedSearchCV
random_search_results.best_score_


# In[77]:


#assign the best set of parameters to final_params
final_params = random_search_results.best_params_
print(final_params)


# In[38]:


#establish the optimized model using the parameters from final_params
create_model(final_params['opt'], final_params['act'], final_params['nl'], final_params['nn'])

#set up an early stopping callback with a patience of 3
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3)

#run the model with 25 epochs, saving the results to history
history = model.fit(X_train, y_train, 
                    epochs = 25,
                    validation_split = 0.2,
                    callbacks = [early_stopping])


# # 6. Evaluate Model

# In[39]:


#plot the training and validation loss over each epoch
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('Model Accuracy by Epoch')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend({'Train', 'Validation'})

plt.savefig('model_acc_latlong.png', dpi=300)


# In[44]:


y_pred = model.predict(X_test)


# In[75]:


#bind y_test and pred into a single df/ndarray and take a random sample to plot
bound_vals = pd.DataFrame(columns= ['predicted', 'actual'])
bound_vals['predicted'] = y_pred
bound_vals['actual'] = y_test

bound_vals_sample = bound_vals.sample(5000)


# In[76]:


#plot the test vals against the predictions
plt.scatter(bound_vals_sample['actual'], bound_vals_sample['predicted'], color = 'darkred', alpha = 0.05)
plt.title('Predicted Values vs. Actual Values')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')

plt.savefig('pred_act_scatter_new.png', dpi=300)


# In[ ]:




