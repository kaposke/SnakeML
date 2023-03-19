
from controllers.RandomSnakeController import RandomSnakeController
from game.Vec2 import Vec2
from simulation.GameManager import GameManager
from renderers.PyGameRenderer import PyGameRenderer
from renderers.TextRenderer import TextRenderer


if __name__ == '__main__':
    gameManager = GameManager(Vec2(20, 20), RandomSnakeController())
    textRenderer = TextRenderer()
    pygameRenderer = PyGameRenderer()

    for i in range(10):
        gameManager.restart()
        while gameManager.is_running:
            gameManager.step()
            textRenderer.render(gameManager.game_observation)
            pygameRenderer.render(gameManager.game_observation)

    pygameRenderer.quit()
