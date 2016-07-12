class Box(object):
    """A 2D bounding box

    Attributes:
        x: x-coordinate of top-left corner
        y: y-coordinate of top-left corner
        width: width of bounding box
        height: height of bounding box

    """
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def grow(self, value=0, top=0, right=0, bottom=0, left=0):
        self.x -= value + left
        self.y -= value + top
        self.width += 2 * value + left + right
        self.height += 2 * value + top + bottom

    def shrink(self, value=0, top=0, right=0, bottom=0, left=0):
        self.grow(-value, -top, -right, -bottom, -left)

    @property
    def extents(self):
        return self.x + self.width, self.y + self.height

    @property
    def area(self):
        return self.width * self.height

    @property
    def indices(self):
        """(slice, slice): indices of the box for slicing from a larger array"""
        return slice(self.y, (self.y + self.height)), slice(self.x, (self.x + self.width))

    def __or__(self, other):
        if isinstance(other, Box):
            x = min(self.x, other.x)
            y = min(self.y, other.y)
            width = max(self.x + self.width, other.x + other.width) - x
            height = max(self.y + self.height, other.y + other.height) - y
            return Box(x, y, width, height)
        else:
            raise TypeError('unsupported operand type(s) for |: \'Box\' and \'int\''.format(type(self), type(other)))

    def __and__(self, other):
        if isinstance(other, Box):
            x = max(self.x, other.x)
            y = max(self.y, other.y)
            width = max(min(self.x + self.width, other.x + other.width) - x, 0)
            height = max(min(self.y + self.height, other.y + other.height) - y, 0)
            return Box(x, y, width, height)
        else:
            raise TypeError('unsupported operand type(s) for |: \'Box\' and \'int\''.format(type(self), type(other)))

    def __bool__(self):
        return self.area > 0

    def __repr__(self):
        return 'Box(x={}, y={}, width={}, height={})'.format(self.x, self.y, self.width, self.height)
