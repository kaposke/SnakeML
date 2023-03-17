
from GameObservation import GameObservation
from Game import Game
from Vec2 import Vec2
from SnakeController import SnakeController


class GameManager:
    def __init__(self, board_size: Vec2, snake_controller: SnakeController):
        self._snake_controller = snake_controller
        self._game = Game(board_size)

    def step(self):
        self._game_observation = self._generate_observation()
        direction = self._snake_controller.pick_direction(
            self._game_observation)
        self._game.snake.direction = direction
        self._game.step()

    def _generate_observation(self):
        return GameObservation(
            self._game.board_size,
            self._game.is_over,
            self._game._apple_position,
            self._game.snake.body,
            self._game.snake.direction,
            self._game.score
        )

    @property
    def is_running(self):
        return not self._game.is_over
