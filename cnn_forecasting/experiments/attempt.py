import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, Flatten, Add
from tensorflow.keras import Model, metrics, utils
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.preprocessing import StandardScaler
import numpy as np


def create_windows(features, targets, lookback, horizon):
    """Turn a timeseries into overlapping windows and categorical labels."""
    X, y = [], []
    end = len(features) - lookback - horizon + 1
    for start in range(end):
        end_idx = start + lookback
        horizon_idx = end_idx + horizon - 1
        X.append(features[start:end_idx])
        y.append(targets[horizon_idx])
    X = np.array(X, dtype=np.float32)
    y = utils.to_categorical(np.array(y, dtype=np.int32))
    return X, y


def make_train_val_split(X, y, train_frac=0.8):
    split = int(len(X) * train_frac)
    return (X[:split], y[:split]), (X[split:], y[split:])


def build_temporal_cnn(
    input_shape,
    num_classes,
    filters=32,
    kernel_size=3,
    stacks=3,
    dilation_base=2,
    dropout_rate=0.1,
):
    inputs = Input(shape=input_shape)

    x = inputs
    dilation = 1
    for _ in range(stacks):
        res = x
        x = Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation, activation="relu")(x)
        x = Dropout(dropout_rate)(x)
        x = Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation, activation="relu")(x)
        res = Conv1D(filters, 1, padding="same")(res)
        x = Add()([x, res])
        dilation *= dilation_base

    x = Flatten()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss=CategoricalCrossentropy(),
        metrics=[metrics.CategoricalAccuracy(name="acc"), metrics.AUC(name="auc")],
    )
    return model


def generate_synthetic_ohlcv(length=1200, seed=7):
    """Lightweight OHLCV generator to keep the example runnable without data files."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.6, size=length)
    trend = np.linspace(0, 5, length)
    close = 100 + np.cumsum(steps + 0.02 * np.sin(np.arange(length) / 15) + trend / length)
    open_ = close + rng.normal(0, 0.3, size=length)
    high = np.maximum(open_, close) + rng.random(length) * 0.5
    low = np.minimum(open_, close) - rng.random(length) * 0.5
    volume = rng.normal(1e6, 5e4, size=length)
    features = np.stack([open_, high, low, close, volume], axis=1)
    direction = (np.diff(close, prepend=close[0]) > 0).astype(np.int32)
    return features, direction


def prepare_data(lookback=32, horizon=1, train_frac=0.8):
    features, direction = generate_synthetic_ohlcv()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    X, y = create_windows(scaled_features, direction, lookback, horizon)
    (X_train, y_train), (X_val, y_val) = make_train_val_split(X, y, train_frac)
    return (X_train, y_train), (X_val, y_val)


def run_experiment(epochs=8, batch_size=32):
    (X_train, y_train), (X_val, y_val) = prepare_data()
    model = build_temporal_cnn(input_shape=X_train.shape[1:], num_classes=y_train.shape[1])
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )
    eval_result = model.evaluate(X_val, y_val, verbose=0)
    return model, history, dict(zip(model.metrics_names, eval_result))


if __name__ == "__main__":
    model, history, metrics_out = run_experiment()
    print("Final validation metrics:", metrics_out)

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

labels = utils.to_categorical(labels, num_classes=3)
