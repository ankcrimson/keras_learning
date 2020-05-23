from keras.preprocessing import sequence

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.callbacks import EarlyStopping
from keras.datasets import imdb
# for pickle error
import numpy as np
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

NUM_WORDS = 6000  # topmost frequent words to consider
SKIP_TOP = 0  # skip top words like the, and, a etc
MAX_REVIEW_LEN = 400  # max number of words


(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=NUM_WORDS, skip_top=SKIP_TOP
)

# print(f"encoded word sequence: {X_train[3]}")
# make word length same by padding
X_train = sequence.pad_sequences(X_train, maxlen=MAX_REVIEW_LEN)
X_test = sequence.pad_sequences(X_test, maxlen=MAX_REVIEW_LEN)
print(f"train shape: {X_train.shape}, test shape: {X_test.shape}")


model = Sequential()
model.add(Embedding(NUM_WORDS, 64, input_length=MAX_REVIEW_LEN))
model.add(LSTM(128))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy"])

# Train
BATCH_SIZE = 24
EPOCHS = 5

cbk = EarlyStopping(monitor="val_acc", mode="max")

model.fit(X_train, y_train, BATCH_SIZE, epochs=EPOCHS,
          validation_data=(X_test, y_test), callbacks=[cbk])
