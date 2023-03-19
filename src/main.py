
from simulation.GameObservation import GameObservation
from simulation.Simulation import Simulation
from game.Vec2 import Vec2
from datetime import datetime
import math
import os
import random
from keras.models import Model, clone_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, concatenate, Reshape
import numpy as np
from joblib import Parallel, delayed
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


board_size = Vec2(20, 20)

# Define the size of the population
population_size = 20
num_generations = 10

# Define the size of the input and output layers
input_size = 5
output_size = 2


def fitness_score(weights):
    network = create_snake_model(board_size)
    network.set_weights(weights)

    simulation = Simulation(board_size)

    scores = []

    for _ in range(5):
        simulation.restart()
        observation = simulation.game_observation
        while simulation.is_running:
            prediction = network.predict(
                preprocess_input(observation), verbose=0)
            simulation.set_input(prediction_to_direction(prediction))
            observation = simulation.step()
        scores.append(observation.score)

    return np.average(scores)


def preprocess_input(observation: GameObservation):
    board = preprocess_observation(observation)
    board = np.expand_dims(board, axis=0)
    direction = preprocess_direction(observation.snake_direction)
    direction = np.expand_dims(direction, axis=0)
    return [board, direction]


def preprocess_observation(observation: GameObservation):
    board_size = observation.board_size
    board = np.zeros((board_size.x, board_size.y), dtype=np.float32)

    # Set snake body
    for segment in observation.snake_body:
        board[segment.x, segment.y] = 1

    # Set snake head
    head_x, head_y = observation.snake_head.x, observation.snake_head.y
    board[head_x, head_y] = 2

    # Set apple position
    apple_x, apple_y = observation.apple_position.x, observation.apple_position.y
    board[apple_x, apple_y] = 3

    return board


def preprocess_direction(direction: Vec2):
    return direction.to_tuple()


def prediction_to_direction(prediction):
    action = np.argmax(prediction)
    if action == 0:
        return Vec2(1, 0)
    elif action == 1:
        return Vec2(-1, 0)
    elif action == 2:
        return Vec2(0, -1)
    elif action == 3:
        return Vec2(0, 1)


def create_snake_model(board_size, num_actions=4):
    # Input for the game state (2D matrix)
    board_input = Input(shape=(board_size.x, board_size.y))
    board_reshaped = Reshape((board_size.x, board_size.y, 1))(board_input)
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(board_reshaped)
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    flatten1 = Flatten()(maxpool2)

    # Input for the snake direction (one-hot encoded vector)
    direction_input = Input(shape=(2,))
    dense1 = Dense(16, activation='relu')(direction_input)

    # Concatenate both inputs
    merged = concatenate([flatten1, dense1])

    # Fully connected layers
    dense2 = Dense(128, activation='relu')(merged)
    output = Dense(num_actions, activation='linear')(dense2)

    # Create the model
    model = Model(inputs=[board_input, direction_input], outputs=output)
    model.compile(loss='mse', optimizer='adam')
    return model


def fitness_scores(parallel, model_weights_list):
    scores = parallel(
        delayed(fitness_score)(model_weights) for model_weights in model_weights_list)
    return scores


def selection(model_weights_list, scores, num_parents):
    selected_indices = np.argsort(scores)[-num_parents:]
    selected_weights = [model_weights_list[i] for i in selected_indices]
    return selected_weights


def crossover(parent_weights, num_offspring):
    offspring_weights = []
    for _ in range(num_offspring):
        parent1_weights, parent2_weights = random.sample(parent_weights, 2)
        crossover_point = random.randint(0, len(parent1_weights) - 1)

        child_weights = parent1_weights[:crossover_point] + \
            parent2_weights[crossover_point:]
        offspring_weights.append(child_weights)
    return offspring_weights


def mutate(model_weights, mutation_rate):
    mutated_weights = []
    for weight in model_weights:
        mutation_mask = np.random.rand(*weight.shape) < mutation_rate
        random_values = np.random.randn(*weight.shape) * mutation_mask
        mutated_weights.append(weight + random_values)
    return mutated_weights


def random_weights(model):
    random_weights = []
    for layer in model.layers:
        layer_weights = []
        for weight in layer.get_weights():
            random_weight = np.random.randn(*weight.shape) * 0.1
            layer_weights.append(random_weight)
        random_weights.extend(layer_weights)
    return random_weights


# Parameters
population_size = 20
num_generations = 100
num_parents = 3
num_offspring = population_size - num_parents
mutation_rate = 0.1

parallel = Parallel(n_jobs=-1)

# Create initial population
model = create_snake_model(board_size)
model_weights_list = [random_weights(model) for _ in range(population_size)]

# Run the genetic algorithm
output_dir = f"output/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)

for generation in range(num_generations):
    print(f"Starting generation {generation + 1}")

    print("Running simulation and evaluating fitness scores")
    # Evaluate fitness scores
    scores = fitness_scores(parallel, model_weights_list)

    # Print the best score in the current generation
    print(f"Generation {generation + 1}, Best score: {max(scores)}")

    print("Selecting top parents")
    # Select the top parents
    parent_weights = selection(model_weights_list, scores, num_parents)

    print("Saving best model")
    # Find the best model in the current generation
    best_score_index = np.argmax(scores)
    best_model_weights = model_weights_list[best_score_index]

    # Save the best model of the current generation
    best_model = create_snake_model(board_size)
    best_model.set_weights(best_model_weights)
    best_model.save(f"{output_dir}/gen{generation}.h5")

    print("Running crossover")
    # Crossover parents to create offspring
    offspring_weights = crossover(parent_weights, num_offspring)

    print("Running mutations")
    # Mutate offspring
    mutated_offspring_weights = [
        mutate(child_weights, mutation_rate) for child_weights in offspring_weights]

    # Update population with parents and mutated offspring
    model_weights_list = parent_weights + mutated_offspring_weights
