import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D

def huber_loss(a, b, in_keras=True):
    error = a - b
    quadratic_term = error*error / 2
    linear_term = abs(error) - 1/2
    use_linear_term = (abs(error) > 1.0)
    if in_keras:
        # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
        use_linear_term = K.cast(use_linear_term, 'float32')
    return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term

class QNetwork:
    def __init__(self, weightsName='TestModelWeights.h5', action_space=4, model=None, lr=0.00025):
        # state inputs to the Q-network
        if model is not None:
            self.model = model
            return
        self.action_space = action_space
        self.model = Sequential()
        self.weightsName=weightsName

        self.model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                         input_shape=(84, 84, 4)))
        self.model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(action_space, activation='linear'))

        self.optimizer = keras.optimizers.RMSprop(
            lr=lr, rho=0.95, epsilon=0.01
        )
        # if os.path.exists('breakout-v0-weights-v0.h5'):
        #     self.loadModel()
        #     print('loaded nn weights from file')

        self.model.compile(loss=huber_loss,
                           optimizer=self.optimizer)

    def copyModel(self):
        copy_model = keras.models.clone_model(self.model)
        copy_model.set_weights(self.model.get_weights())
        return QNetwork(model=copy_model, action_space=self.action_space)

    def saveModel(self):
        self.model.save_weights(self.weightsName)

    def loadModel(self):
        self.model.load_weights(self.weightsName)

    def predict(self, state):
        prediction = self.model.predict(np.array([state]))[0]
        return prediction


    def fit(self, x, y):
        self.model.fit(x=x, y=y, shuffle=False, verbose=False)

    def setWeights(self, weights):
        self.model.set_weights(weights)

    def getWeights(self):
        return self.model.get_weights()

