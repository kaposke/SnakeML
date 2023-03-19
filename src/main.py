
from controllers.RandomSnakeController import RandomSnakeController
from game.Vec2 import Vec2
from simulation.GameManager import GameManager
from renderers.TextRenderer import TextRenderer


if __name__ == '__main__':
    gameManager = GameManager(Vec2(20, 20), RandomSnakeController())
    renderer = TextRenderer()

    while gameManager.is_running:
        gameManager.step()
        renderer.render(gameManager.game_observation)
