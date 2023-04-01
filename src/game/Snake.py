from .Vec2 import Vec2


class Snake:

    def __init__(self, body):
        self._body: list[Vec2] = body
        self._direction: Vec2 = Vec2(1, 0)

    def step(self):
        print(self.direction)
        # Move body
        body_len = len(self.body)
        for i in reversed(range(1, body_len)):
            self.body[i] = self.body[i-1]

        # Move head
        self.body[0] += self.direction

    def grow(self):
        self._body.append(self.body[-1])

    @property
    def body(self):
        return self._body

    @property
    def head(self):
        return self._body[0]

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, new_dir: Vec2):
        if abs(new_dir.x + new_dir.y) > 1:
            raise Exception('Invalid direction provided')

        dir_sum = self.direction + new_dir
        if dir_sum.x == 0 and dir_sum.y == 0:
            return

        self._direction = new_dir

    @property
    def is_overlapping_itself(self):
        for i in range(0, len(self.body)):
            for j in range(i + 1, len(self.body)):
                if self.body[i] == self.body[j]:
                    return True
        return False
