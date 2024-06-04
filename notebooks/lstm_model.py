from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


class LSTMModel:
    def __init__(self, num_labels):
        self.tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')

        self.model = Sequential()
        self.model.add(Embedding(input_dim=5000, output_dim=64, input_length=100))
        self.model.add(LSTM(128, return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(64))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(num_labels, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
