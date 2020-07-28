# Bike Availability in the SFMTA Bikeshare System
## Neural Network Machine Learning for Regression

#### Motivation
One of the benefits of the advancement of dockless bikeshare programs is that no individual station can be void of bikes to check out, or full of bikes, preventing a check in. However, dock-based bikeshare systems are still commonplace throughout the country, and due to the large capital investment they required, will liekly remain so for some time. San Francisco MTA runs one of the largest dock-based bikeshare systems in the country, and has made data about its usage [publicaly available](https://www.sfmta.com/getting-around/bike/bike-share#Bikeshare%20Data). 

Joining the bike usage data with weather data from [Kaggle](https://www.kaggle.com/benhamner/sf-bay-area-bike-share), I was able to generate a useful set of inputs that could help determine the likely number of bikes available at a given dock, based on the weather, location, day, and time.

#### Methods
The dataset used for this project was very large, and scaled up rapidy as additional varibles were created.

I managed the import of data by using SQLite, and randomly sampling the import to speed up computation. The status file wasn particularly large, so that import was handled in chunks, and the chunks were sampled individually. 

Disparate data sets were merged in Python, using the Pandas package. The weather dataset was pretty messy, so missing and duplicate data had to be dealt with using a combination of methods. 

Time series data was handled by breaking the datetimes into constituent parts, and analyzing those parts as discrete values. 

The exploratory data analysis showed that the primary input values had individual predictive power, suggesting that they would be even better in concert. I constructed a sequential model in Keras, and used RandomizedSearchCV to tune the hyperparameters to their optimal values. 

#### Outcomes
The final model had a mean aboslute error of 2.2, which was much lower than the standard deviation of 4.0. This suggested a strongly predictive model that produced statistically robust predictions.

This system could be used as a stand-alone, or it could possibly be integrated into a trip planning system to help direct users to docks where bikes are more likely to be available at the time of the planned trip.
