import time
from game.Vec2 import Vec2
from simulation.Simulation import Simulation
from renderers.TextRenderer import TextRenderer
from renderers.PyGameRenderer import PyGameRenderer

board_size = Vec2(20, 20)
simulation = Simulation(board_size)

observation = simulation.game_observation

pygameRenderer = PyGameRenderer()
pygameRenderer.render(observation)

for _ in range(100):
    simulation.restart()
    while simulation.is_running:
        simulation.set_input(Vec2(0, -1))
        observation = simulation.step()
        pygameRenderer.render(observation)

pygameRenderer.quit()

