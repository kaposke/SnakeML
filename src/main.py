
from RandomSnakeController import RandomSnakeController
from Vec2 import Vec2
from GameManager import GameManager


if __name__ == '__main__':
    gameManager = GameManager(Vec2(20, 20), RandomSnakeController())

    while gameManager.is_running:
        gameManager.step()
