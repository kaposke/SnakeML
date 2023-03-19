import pygame

from simulation.GameObservation import GameObservation


class PyGameRenderer:
    def __init__(self, caption='PyGame Renderer', screen_size=(800, 800)):
        pygame.init()

        self._screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption(caption)
        self._font = pygame.font.SysFont('arial black', 128)

        self._running = True

        self._board_size = None
        self._cell_size = None

    def render(self, game_observation: GameObservation):
        if not self._running:
            return

        if self._board_size is None:
            self._board_size = game_observation.board_size
            self._cell_size = self._calculate_cell_size(
                self._screen.get_size(), self._board_size.to_tuple())

        self._poll_events()

        self._screen.fill("black")

        self._draw_score(game_observation.score)

        self._draw_snake(game_observation.snake_body)

        self._draw_apple(game_observation.apple_position)

        pygame.display.flip()

    def _poll_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False

    def _draw_score(self, score):
        text_surface = self._font.render(
            str(score), True, (50, 50, 50))

        text_rect = text_surface.get_rect()

        text_rect.center = (self._screen.get_width() // 2,
                            self._screen.get_height() // 2)

        self._screen.blit(text_surface, text_rect)

    def _draw_snake(self, snake_body):
        cellW, cellH = self._cell_size

        for cell in snake_body:
            pygame.draw.rect(self._screen, "white",
                             (cell.x * cellW, cell.y * cellH, cellW, cellH))

    def _draw_apple(self, apple_position):
        cellW, cellH = self._cell_size

        pygame.draw.ellipse(self._screen, "red",
                            (apple_position.x * cellW, apple_position.y * cellH, cellW, cellH))

    def _calculate_cell_size(self, screen_size, board_size):
        screenW, screenH = screen_size
        boardW, boardH = board_size
        return (screenW / boardW, screenH / boardH)

    def quit(self):
        pygame.quit()
