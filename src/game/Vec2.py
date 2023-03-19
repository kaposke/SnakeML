from dataclasses import dataclass
import math


@dataclass
class Vec2:
    def __init__(self, x: int = 0, y: int = 0):
        self._x = x
        self._y = y

    def to_tuple(self):
        return (self.x, self.y)

    @property
    def x(self) -> int:
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self) -> int:
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, other) -> int:
        return Vec2(self.x * other.x, self.y * other.y)

    def __truediv__(self, other):
        return Vec2(self.x / other.x, self.y / other.y)

    def __abs__(self) -> int:
        return math.sqrt(self.x**2 + self.y**2)

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y

    def __str__(self) -> str:
        return '(%g, %g)' % (self.x, self.y)

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)
