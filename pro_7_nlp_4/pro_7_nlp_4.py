#-------------------------------- EE 6530 ---------------------------------
#| Project #7 -  Natural language processing using ANN                    |
#--------------------------------------------------------------------------
#
# Instructor            : Prof. Uyar
#
# Student 1 Name        : James Bond
# Student 1 CCNY email  : jbond007
# Student 1 Log In Name : ee6530_007
# Student 2 Name        : 
# Student 2 CCNY email  :
# Student 2 Log In Name : 
# Student 3 Name        :
# Student 2 CCNY email  :
# Student 3 Log In Name :
# --------------------------------------------------------------------------
# | I UNDERSTAND THAT COPYING PROGRAMS FROM OTHERS WILL BE DEALT           |
# | WITH DISCIPLINARY RULES OF CCNY.                                       |
# --------------------------------------------------------------------------

# Using tweets from the FDA, predict the movement of Moderna stocks with 
# the help of an ANN.
# Define your own key words. 
# The days where no data was present are given in arrays called missing_days 
# and invalid days.
import datetime 
from datetime import date
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import re

#%% Function to remove punctuation and web links from tweets
# @\S+|https?://\S+ - matches either a substring which starts with @ 
# and contains non-whitespace characters \S+ OR a link(url) which 
# starts with http(s)://

def clean_tweet(tweet):
    return ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', 
                ' ', tweet).split())

def str_to_date(string):
    return datetime.datetime.strptime(string, '%Y-%m-%d').date()

def print_weights(weights):
    # weights = model.get_weights();
    print('\n******* WEIGHTS OF ANN *******\n') 
    for i in range(int(len(weights)/2)):
        print('Weights W%d:\n' %(i), weights[i*2])
        print('Bias b%d:\n' %(i), weights[(i*2)+1])
        
def normalize_column(dataframe, col_name):
    maximum = max(dataframe[col_name])
    minimum = min(dataframe[col_name])
    dataframe[col_name] = (dataframe[col_name] - minimum)/(maximum-minimum)
    return dataframe[col_name]

#%% Load in file, remove punctuation and weblinks from all tweets and then 
#   perform sentiment analysis.  
print('\n\n********** CLEANING TWEETS **********\n\n')
filename = "FDA _tweets.csv"
df = pd.read_csv(filename)
df["Clean Tweet"] = df['text'].apply(lambda x: clean_tweet(x))
df['created_at'] = pd.to_datetime(df['created_at']).dt.date

#%% Data cleaning and processing  
# Check if the tweet contains any keywords - count the number of times key words are used in tweets 
print('\n\n********** IDENTIFYING KEY WORDS **********\n\n')
key_words = [None]
 
# list of invalid days: including federal holidays, days for when we have no data 
invalid_days = ['2020-07-04','2020-07-05','2020-07-11','2020-07-12','2020-07-19'
                ,'2020-07-26','2020-08-16']
missing_days = ['2020-07-04','2020-07-05','2020-07-11','2020-07-12','2020-07-19'
                ,'2020-07-26','2020-08-16']
noof_missing_days = len(missing_days)
invalid_days = list(map(lambda x:str_to_date(x),invalid_days))

# if the tweets are in invalid days, move them to the next day for prediction.
        
# Move tweets sent on a saturday or sunday to the nearest Monday.
# When using the weekday() function -> 0 is Monday and 6 is Sunday 

# group the data by date - aggregate all keywords used and sum them up.
# for example, the tweets from the day mention 'dollar' multiple times, add
#   them together. 

#%% Stock prices
filename = "Moderna_vaccine_stocks.csv"
moderna_df = pd.read_csv(filename)
# return at close each day
moderna_df['percent change'] = moderna_df['Close'].pct_change() 

missing_days = list(map(lambda x:str_to_date(x), missing_days))
for missing_day in missing_days: 
    grouped_df = grouped_df[grouped_df['created_at'] != missing_day]

# %% neural network - IN PROGRESS
# create the input data  
grouped_df['day_opening'] = moderna_df['Open']
grouped_df['units_traded'] = moderna_df['Volume']
days_to_predict = 0.35*len(grouped_df)+noof_missing_days

# before putting any data into NN, we normalize the data (between 0 and 1)
# Normailize only the open and volume traded (beacuse they are large compared
# to the other input values)

# create the input and output arrays
features = ['noof_keywords', 'day_opening', 'units_traded']

neural_net = tf.keras.Sequential()

# Define the input layer

## add another hidden layer with 4 neurons to the NN

## add another hidden layer with 8 neurons to the NN

## add an output layer with a single output (percent change)

neural_net.compile(optimizer='adam', loss='mean_absolute_error')

## train the ANN model using 1200 iterations
print('\n\n********** Begin ANN training **********\n\n')

weights = neural_net.get_weights()
print_weights(weights)
print('\n\n********** ANN training complete **********\n\n')

# %%  make predictions
noof_correct_movement = 0
noof_predictions = 0
diffs = []
input_features = []
print('\n\n********** ANN PREDICTIONS **********\n\n')
for i in range(int(days_to_predict), int(noof_missing_days),-1):
    input_features.clear()
    noof_predictions += 1
    actual_change = moderna_df.iloc[(-1*i)+1]['percent change']
    for feature in features:
        input_features.append(grouped_df.iloc[-1*i][feature])
    predicted_change = neural_net.predict(np.array([input_features]))[0,0]
    
    print(f"The predicted change for {grouped_df.iloc[(-1*i)+1]['created_at']}",
          end='') 
    print(f" was: {predicted_change:.5f}")
    print(f"Actual change was for {moderna_df.iloc[(-1*i)+1]['Date']} was: ",
          end='')
    print(f"{round(actual_change,4)}\n")
    
    if (predicted_change * actual_change) > 0:
        diffs.append(math.fabs(predicted_change - actual_change))
        noof_correct_movement += 1

percent_correct = (noof_correct_movement/noof_predictions) * 100
print(f"ANN was correct in predicting the movement ", end = '')
print(f"{((noof_correct_movement/noof_predictions) * 100):.2f}% of the time in",
      end='')
print(f" {noof_predictions} predictions.")
average_diff = round(sum(diffs)/len(diffs),5)
print(f"The average error of the correct predictions were", end='')
print(f" {average_diff* 100.0:.1f} %")

with open('output.txt','a') as output_file:
    output_file.write(f'{date.today()} percentage of times correct prediction:')
    output_file.write(f' {round(percent_correct,1)} % ')
    output_file.write(f' with error {average_diff* 100.0:.1f}%\n')

