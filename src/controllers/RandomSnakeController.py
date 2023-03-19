import random
from game.Vec2 import Vec2
from simulation.GameObservation import GameObservation
from simulation.SnakeController import SnakeController


class RandomSnakeController(SnakeController):

    possible_directions = [
        Vec2(1, 0),
        Vec2(0, 1),
        Vec2(-1, 0),
        Vec2(0, -1),
    ]

    def pick_direction(self, observation: GameObservation):
        return random.choice(self.possible_directions)
