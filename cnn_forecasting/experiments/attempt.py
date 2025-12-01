import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dense, Lambda, Flatten, Concatenate
from tensorflow.keras import Model
from tensorflow.keras import metrics
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import utils
from sklearn.preprocessing import StandardScaler
import numpy as np
import math

inputs = Input(shape=(15, 5))
feature_extraction = Conv1D(30, 4, activation='relu')(inputs)
long_term = Lambda( lambda x: tf.split(x, num_or_size_splits=3, axis=1)[0])(feature_extraction)
mid_term = Lambda( lambda x: tf.split(x, num_or_size_splits=3, axis=1)[1])(feature_extraction)
short_term = Lambda( lambda x: tf.split(x, num_or_size_splits=3, axis=1)[2])(feature_extraction)

long_term_conv = Conv1D(1, 1, activation='relu')(long_term)
mid_term_conv = Conv1D(1, 1, activation='relu')(mid_term)
short_term_conv = Conv1D(1, 1, activation='relu')(short_term)

combined = Concatenate(axis=1)([long_term_conv, mid_term_conv, short_term_conv])
flattened = Flatten()(combined)

outputs = Dense(3, activation='softmax')(flattened)

input_vars = ['open', 'high', 'low', 'close', 'volume']

class Direction:
    UP = 0
    DOWN = 1
    STATIONARY = 2

rolling_avg_window_size = 5

shift = -(rolling_avg_window_size-1)

stationary_threshold = .0001

scaler = StandardScaler()

df['close_avg'] = df['close'].rolling(window=rolling_avg_window_size).mean().shift(shift)
df['close_avg_change_pct'] = (df['close_avg'] - df['close']) / df['close']

def label_data(row):
    if row['close_avg_change_pct'] > stationary_threshold:
        return Direction.UP
    elif row['close_avg_change_pct'] < -stationary_threshold:
        return Direction.DOWN
    else:
        return Direction.STATIONARY

df['movement_labels'] = df.apply(label_data, axis=1)

data = []
labels = []

for i in range(len(df)-self.n_tsteps+1+shift):
    label = df['movement_labels'].iloc[i+self.n_tsteps-1]
    data.append(df[input_vars].iloc[i:i+self.n_tsteps].values)
    labels.append(label)

data = np.array(data)

dim1, dim2, dim3 = data.shape
data = data.reshape(dim1*dim2, dim3)
data = scaler.fit_transform(data)
data = data.reshape(dim1, dim2, dim3)