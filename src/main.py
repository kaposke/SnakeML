
from RandomSnakeController import RandomSnakeController
from Vec2 import Vec2
from GameManager import GameManager
from TextRenderer import TextRenderer


if __name__ == '__main__':
    gameManager = GameManager(Vec2(20, 20), RandomSnakeController())
    renderer = TextRenderer()

    while gameManager.is_running:
        gameManager.step()
        renderer.render(gameManager.game_observation)
