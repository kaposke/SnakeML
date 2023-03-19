from game.Game import Game
from game.Vec2 import Vec2
from .GameObservation import GameObservation


class Simulation:
    def __init__(self, board_size: Vec2):
        self._game = Game(board_size)
        self._game_observation = self._generate_observation()

        self._input = None

    def restart(self):
        self._game.restart()

    def set_input(self, direction: Vec2):
        self._input = direction

    def step(self):
        if self._input is not None:
            self._game.set_snake_direction(self._input)
            self._input = None

        self._game.step()
        self._game_observation = self._generate_observation()
        return self.game_observation

    def _generate_observation(self):
        return GameObservation(
            self._game.board_size,
            self._game.is_over,
            self._game.apple_position,
            self._game.snake_body,
            self._game.snake_head,
            self._game.snake_direction,
            self._game.score
        )

    @property
    def is_running(self):
        return not self._game.is_over

    @property
    def game_observation(self):
        return self._game_observation
