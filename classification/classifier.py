import os

import numpy as np
from keras.initializers import TruncatedNormal
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD

from vector import vectorLoader


class KerasClassifier:
    def __init__(self):
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.load_in_out()
        self.train_model(7,20,45)

    def load_in_out(self):
        data = vectorLoader.load_vectors()
        feature_vectors = np.asarray(data[0])
        output_vectors = np.asarray(data[1])

        train_limit = int(0.85 * len(feature_vectors))
        self.x_train = feature_vectors[:train_limit]
        self.y_train = output_vectors[:train_limit]

        self.x_test = feature_vectors[train_limit:]
        self.y_test = output_vectors[train_limit:]

    def train_model(self, n, e, bs):
        # TODO: change dimensions
        output_length = 0
        input_length = 0
        self.model = Sequential([
            Dense(n, activation='relu', input_dim=input_length, name='internal', kernel_initializer=TruncatedNormal(seed=17)),
            Dense(output_length, activation='softmax', kernel_initializer=TruncatedNormal(seed=31))
        ])

        sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])

        self.model.fit(self.x_train, self.y_train, epochs=e, batch_size=bs)
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print("The accuracy of this model is", score[1])

        # serialize model to JSON
        model_json = self.model.to_json()
        mode_path = "{}/../model/modelCBOW.json".format(self.path)
        with open(mode_path, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        weights_path = "{}/../model/modelCBOW.h5".format(self.path)
        self.model.save_weights(weights_path)

        return score[1]

if __name__ == '__main__':
    KerasClassifier()
