
from game.Vec2 import Vec2
from .Simulation import Simulation
from .SnakeController import SnakeController


class ControlledSimulation:
    def __init__(self, board_size: Vec2, snake_controller: SnakeController):
        self._snake_controller = snake_controller
        self._simulation = Simulation(board_size)
        self._game_observation = self._simulation.game_observation

    def restart(self):
        self._simulation.restart()
        self._game_observation = self._simulation.game_observation

    def step(self):
        direction = self._snake_controller.pick_direction(
            self._game_observation)
        self._simulation.set_input(direction)
        self._game_observation = self._simulation.step()

    @property
    def is_running(self):
        return self._simulation.is_running

    @property
    def game_observation(self):
        return self._game_observation
