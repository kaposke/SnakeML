# %%
# Import necessary libraries
import numpy as np
import random
from collections import deque

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from game.Vec2 import Vec2
from simulation.Simulation import Simulation
from simulation.GameObservation import GameObservation
from renderers.PyGameRenderer import PyGameRenderer

# %%
def build_model(input_shape, input_size, output_shape, learning_rate):
    inputs = layers.Input(shape=input_shape)
    flatten = layers.Flatten()(inputs)
    layer1 = layers.Dense((input_size * 3 / 2), activation="relu", name="layer1")(
        flatten
    )
    layer2 = layers.Dense((input_size * 2 / 3), activation="relu", name="layer2")(
        layer1
    )
    Qactions = layers.Dense(output_shape, name="Qactions")(layer2)

    model = keras.Model(inputs=inputs, outputs=Qactions)

    model.compile(optimizer=Adam(learnin_rate=learning_rate), loss="mean_squared_error")

    return model


# %%
directions = [
    Vec2(1, 0),
    Vec2(0, 1),
    Vec2(-1, 0),
    Vec2(0, -1),
]


def get_action_from_index(index: int):
    return directions[index]


def generate_state(observation: GameObservation):
    state = np.zeros((*observation.board_size.to_tuple(), 3))

    state[observation.apple_position.x][observation.apple_position.y][0] = 1

    for cell in observation.snake_body:
        if (
            cell.x >= 0
            and cell.x < observation.board_size.x
            and cell.y >= 0
            and cell.y < observation.board_size.y
        ):
            state[cell.x][cell.y][1] = 1

    if not observation.is_game_over:
        state[observation.snake_head.x][observation.snake_head.y][2] = 1

    return state


# %%
import os
from datetime import datetime

output_dir = f"output/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)

# %%
# Define hyperparameters
EPISODES = 10000
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9999
LEARNING_RATE = 0.001

MAX_STEPS = 1000

# %%
# Initialize environment and DQN models
board_size = Vec2(20, 20)

simulation = Simulation(board_size)
# renderer = PyGameRenderer()
sample_state = generate_state(simulation.game_observation)
state_shape = sample_state.shape
state_size = sample_state.size
action_size = 4

q_model = build_model(state_shape, state_size, action_size, LEARNING_RATE)
target_model = build_model(state_shape, state_size, action_size, LEARNING_RATE)

# Initialize memory
memory = deque(maxlen=100000)

# %%
q_model = keras.models.load_model("output/20230401_214333/episode110.h5")
target_model.set_weights(q_model.weights)

# %%
# Define function to update target model
def update_target_model():
    target_model.set_weights(q_model.get_weights())


# Define function to train the model
def train_dqn():
    epsilon = EPSILON_START

    for episode in range(EPISODES):
        simulation.restart()
        observation = simulation.game_observation
        state = generate_state(observation)
        done = False
        total_reward = 0

        step = 0

        while not done and step < MAX_STEPS:
            step += 1

            # renderer.render(observation)

            # Choose action using epsilon-greedy policy
            if np.random.rand() <= epsilon:
                action = random.randint(0, action_size - 1)
            else:
                action = np.argmax(
                    q_model.predict(np.expand_dims(state, axis=0), verbose=0)
                )

            # Take action and observe next state, reward, and done
            previous_score = observation.score

            simulation.set_input(get_action_from_index(action))
            observation = simulation.step()

            next_state = generate_state(observation)
            reward = observation.score - previous_score
            done = observation.is_game_over

            total_reward += reward

            # Store experience in memory
            memory.append((state, action, reward, next_state, done))

            # Update current state
            state = next_state

            # Train the model if memory has enough samples
            if len(memory) >= BATCH_SIZE:
                minibatch = random.sample(memory, BATCH_SIZE)
                update_model(minibatch)

                # Decay epsilon
                epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)

            if done:
                print(
                    "Episode: {}, Total Reward: {}, Epsilon: {:.2}".format(
                        episode, total_reward, epsilon
                    )
                )
                break

        # Update target model every few episodes
        if episode % 10 == 0:
            print("Updating target model")
            update_target_model()
            q_model.save(f"{output_dir}/episode{episode}.h5")


# Define function to update the Q model
def update_model(minibatch):
    states = np.array([sample[0] for sample in minibatch])
    actions = np.array([sample[1] for sample in minibatch])
    rewards = np.array([sample[2] for sample in minibatch])
    next_states = np.array([sample[3] for sample in minibatch])
    dones = np.array([sample[4] for sample in minibatch])

    # Get Q-values of next states from target model
    next_q_values = target_model.predict(next_states, verbose=0)
    target_q_values = rewards + (1 - dones) * GAMMA * np.max(next_q_values, axis=1)

    # Update Q-values of current states based on chosen actions
    current_q_values = q_model.predict(states, verbose=0)
    for i in range(BATCH_SIZE):
        current_q_values[i][actions[i]] = target_q_values[i]

    # Train the Q model with updated Q-values
    q_model.fit(states, current_q_values, verbose=0)


# %%
# Train the deep Q-learning model
train_dqn()
