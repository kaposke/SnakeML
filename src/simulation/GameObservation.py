class GameObservation:

    def __init__(self, board_size, is_game_over, apple_position, snake_body, snake_head, snake_direction, score):
        self._board_size = board_size
        self._is_game_over = is_game_over
        self._apple_position = apple_position
        self._snake_body = snake_body
        self._snake_head = snake_head
        self._snake_direction = snake_direction
        self._score = score

    @property
    def board_size(self):
        return self._board_size

    @property
    def is_game_over(self):
        return self._is_game_over

    @property
    def apple_position(self):
        return self._apple_position

    @property
    def snake_body(self):
        return self._snake_body

    @property
    def snake_head(self):
        return self._snake_head

    @property
    def snake_direction(self):
        return self._snake_direction

    @property
    def score(self):
        return self._score

    def __str__(self):
        strs = [
            f'Board size: {self.board_size}',
            f'Is Over: {self.is_game_over}',
            f'Score: {self.score}',
            f'Apple Position: {self.apple_position}',
            f'Snake Body: {", ".join([str(c) for c in self.snake_body])}',
            f'Snake Direction: {self.snake_direction}'
        ]
        return '\n'.join(strs)
