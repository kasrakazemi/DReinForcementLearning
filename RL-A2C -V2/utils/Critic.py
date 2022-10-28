######################## import libs ########################
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
import numpy as np
#############################################################


class Critic:

    def __init__(self, state_dim):

        self.state_dim = state_dim
        self.model = self.create_model()
        #self.opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    def create_model(self):
        
        model =  tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    # def compute_loss(self, v_pred, td_targets):

    #     mse = tf.keras.losses.MeanSquaredError()
    #     return mse(td_targets, v_pred)

    def train(self, memory,model):

        states= [item[0] for item in memory]
        states= [np.reshape(i,(1, np.shape(i)[0])) for i in states]
        td_targets= [item[4] for item in memory]

        self.Critic.fit(states, td_targets, epochs=1, verbose=0)
        # for state,action,reward,next_state,td_target,advantage in memory:

        #     with tf.GradientTape() as tape:
        #         v_pred = model(np.reshape(state,(1, np.shape(state)[0])), training=True)
        #         #assert v_pred.shape == td_targets.shape
        #         loss = self.compute_loss(v_pred, tf.stop_gradient(td_target))

        #     grads = tape.gradient(loss, model.trainable_variables)
        #     self.opt.apply_gradients(zip(grads, model.trainable_variables))
        
        return model
