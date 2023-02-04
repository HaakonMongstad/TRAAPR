import tensorflow as tf
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

VISION_SIZE = 5

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    """

    model = tf.keras.models.Sequential(
        [
            #Convolutional layer
            tf.keras.layers.Dense(50, input_shape=(VISION_SIZE*VISION_SIZE,), activation = "relu"),

            tf.keras.layers.Dense(50, activation='relu'),

            tf.keras.layers.Dense(50, activation='relu'),

            tf.keras.layers.Dense(50, activation='relu'),

            tf.keras.layers.Dense(1, activation='linear')
        ]
    )

    return model

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn