from vision import Box
from functools import total_ordering


@total_ordering
class Ruler(object):
    """A ruler from an image

    This class describes the rulers position in an image, as well as defining quantities that assess its
    suitability to represent an actual ruler and the parameters of that ruler

    Attributes:
        bounds: A Box object describing the bounding box of the ruler in its image
        hspace: An array of the bins resulting from the Hough transform on the cropped image of the ruler
        angles: An array of the angle bins used for the Hough transform :py:attr:`hspace`
        distances: An array of the distance bins used for the Hough transform :py:attr:`hspace`
        score: Measure of how well this ruler fits the expected pattern of a ruler
        angle_index: Index corresponding to the :py:attr:`angles`
            array of the angle of the graduations in the image
        graduations: List of the size of the gaps between different sized graduations, in ascending order
        separation: Distance in *pixels* between the smallest graduations

    """
    def __init__(self, x, y, width, height):
        self.bounds = Box(x, y, width, height)

        self.hspace = None
        self.angles = None
        self.distances = None

        self.score = None
        self.angle_index = None

        self.graduations = []
        self.separation = 0

    @classmethod
    def from_box(cls, box):
        return cls(box.x, box.y, box.width, box.height)

    @property
    def indices(self):
        """(slice, slice): indices of the ruler in the image"""
        return self.bounds.indices

    def __lt__(self, other):
        return self.score < other.score

    def __eq__(self, other):
        return self.score == other.score
