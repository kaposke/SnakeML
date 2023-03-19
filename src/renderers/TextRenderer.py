

from simulation.GameObservation import GameObservation
from game.Vec2 import Vec2


class TextRenderer:
    def __init__(self):
        pass

    def render(self, game_observation: GameObservation):
        score = game_observation.score
        board_size = game_observation.board_size
        snake_body = game_observation.snake_body
        apple_position = game_observation.apple_position

        chars = [f'Score: {score}\n']

        for y in range(0, board_size.y):
            if y == 0:
                chars.append('┏')
                chars += ['━━' for _ in range(board_size.x)]
                chars.append('┓\n')
            chars.append('┃')

            for x in range(0, board_size.x):
                cell = Vec2(x, y)
                if cell == apple_position:
                    chars.append('▓▓')
                    continue

                if cell in snake_body:
                    chars.append('██')
                    continue

                chars.append('░░')

            chars.append('┃')
            chars.append('\n')

            if y == board_size.y - 1:
                chars.append('┗')
                chars += ['━━' for _ in range(board_size.x)]
                chars.append('┛\n')

        final_string = ''.join(chars)

        print(final_string)
