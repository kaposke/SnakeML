import math
import random
from .Vec2 import Vec2
from .Snake import Snake


class Game:

    MIN_BOARD_SIZE = 5

    def __init__(self, board_size: Vec2):
        if board_size.x < self.MIN_BOARD_SIZE or board_size.y < self.MIN_BOARD_SIZE:
            raise Exception(
                f'Minimum allowed board size is {self.MIN_BOARD_SIZE}')

        self._board_size = board_size
        self._setup()

    def _setup(self):
        self._score = 0
        self._initialize_snake()
        self._randomize_apple()
        self._over = False

    def _randomize_apple(self):
        self._apple_position = Vec2(random.randint(
            0, self._board_size.x - 1), random.randint(0, self._board_size.y - 1))

    def _initialize_snake(self):
        center = Vec2(math.floor(self._board_size.x / 2),
                      math.floor(self._board_size.y / 2))

        body = [center, center + Vec2(-1, 0), center + Vec2(-2, 0)]

        self._snake = Snake(body)

    def restart(self):
        self._setup()

    def step(self):
        self._snake.step()
        if self._snake.is_overlapping_itself:
            self.end_game()

        if self._is_snake_head_out_of_board:
            self.end_game()

        if self._is_snake_head_over_apple:
            self._snake.grow()
            self._randomize_apple()
            self._score += 1

    def end_game(self):
        self._over = True

    def set_snake_direction(self, direction: Vec2):
        self._snake.direction = direction

    @property
    def _is_snake_head_out_of_board(self):
        snake_head = self._snake.head
        return snake_head.x < 0 or snake_head.x > self._board_size.x - 1 \
            or snake_head.y < 0 or snake_head.y > self._board_size.y - 1

    @property
    def _is_snake_head_over_apple(self):
        return self._apple_position == self._snake.head

    @property
    def board_size(self):
        return self._board_size

    @property
    def is_over(self):
        return self._over

    @property
    def score(self):
        return self._score

    @property
    def snake_body(self):
        return self._snake.body

    @property
    def snake_direction(self):
        return self._snake.direction

    @property
    def apple_position(self):
        return self._apple_position
