###################### import libs #####################
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
import numpy as np
########################################################

class Actor:

    def __init__(self, state_dim, action_dim):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.create_model()
        #self.opt = tf.keras.optimizers.Adam(learning_rate=0.005)

    def create_model(self):
        
        model = tf.keras.Sequential([
            Input((self.state_dim),),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(self.action_dim, activation='softmax')
        ])
        model.compile(loss='SparseCategoricalCrossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.005))

        return model


    # def compute_loss(self, actions, logits, advantages):

    #     ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(
    #         from_logits=False)
    #     actions = tf.cast(actions, tf.int32)
    #     policy_loss = ce_loss(
    #         actions, logits, sample_weight=tf.stop_gradient(advantages))
				
    #     return policy_loss

    def train(self,memory,model):

        states= [item[0] for item in memory]
        states= [np.reshape(i,(1, np.shape(i)[0])) for i in states]
        actions= [np.reshape(item[1], [1, 1]) for item in memory]
        #actions= [np.array(item[1]) for item in memory]
        advantages= [item[5] for item in memory]

        model.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0)
        # for state,action,reward,next_state,td_target,advantage in memory:
           
        #     with tf.GradientTape() as tape:
        #         logits = model(np.reshape(state,(1, np.shape(state)[0])), training=True)
        #         loss = self.compute_loss(

        #             action, logits, advantage)

        #     grads = tape.gradient(loss, model.trainable_variables)
        #     self.opt.apply_gradients(zip(grads, model.trainable_variables))

        return model



