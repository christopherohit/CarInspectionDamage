import math


# It creates a class called Point, which has two attributes, x and y.
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y



def GetAreaOfPolyGon(points_x, points_y):
    """
    The function takes in a list of x and y coordinates and returns the area of the polygon formed by
    the points
    
    :param points_x: a list of x coordinates of the polygon's vertices
    :param points_y: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0
    """
    points = []
    for index in range(len(points_x)):
        points.append(Point(points_x[index], points_y[index]))
    area = 0
    if (len(points) < 3):
        raise Exception("error")

    p1 = points[0]
    for i in range(1, len(points) - 1):
        p2 = points[1]
        p3 = points[2]

        vecp1p2 = Point(p2.x - p1.x, p2.y - p1.y)
        vecp2p3 = Point(p3.x - p2.x, p3.y - p2.y)

        vecMult = vecp1p2.x * vecp2p3.y - vecp1p2.y * vecp2p3.x
        sign = 0
        if (vecMult > 0):
            sign = 1
        elif (vecMult < 0):
            sign = -1

        triArea = GetAreaOfTriangle(p1, p2, p3) * sign
        area += triArea
    return abs(area)


def GetAreaOfTriangle(p1, p2, p3):
    """
    The function takes three points as input and returns the area of the triangle formed by the three
    points
    
    :param p1: The first point of the triangle
    :param p2: (x, y)
    :param p3: (x, y)
    :return: The area of the triangle.
    """
    area = 0
    p1p2 = GetLineLength(p1, p2)
    p2p3 = GetLineLength(p2, p3)
    p3p1 = GetLineLength(p3, p1)
    s = (p1p2 + p2p3 + p3p1) / 2
    area = s * (s - p1p2) * (s - p2p3) * (s - p3p1)
    area = math.sqrt(area)
    return area


def GetLineLength(p1, p2):
    """
    It calculates the length of a line between two points.
    
    :param p1: The first point
    :param p2: The point that you want to move to
    :return: The length of the line between the two points.
    """
    length = math.pow((p1.x - p2.x), 2) + math.pow((p1.y - p2.y), 2)
    length = math.sqrt(length)
    return length


def main():
    """
    It takes a list of points, and returns the area of the polygon they form
    """
    points = []
    # x=[0.02,0.04,0.05,0.07,0.10,0.12,0.15,0.18,0.21,0.24,0.28,0.32,0.37,0.42,0.46,0.52,0.57,0.62,0.68,0.74,0.80,0.86,0.92,0.99,1.06,1.12,1.19,1.26,1.33,1.40,1.48,1.68,1.75,1.82,1.88,1.95,2.01,2.08,2.15,2.21,2.28,2.35,2.41,2.48,2.55,2.61,2.68,2.75,2.81,2.88,2.95,3.01,3.08,3.15,3.21,3.27,3.34,3.39,3.46,3.51,3.58,3.64,3.69,3.75,3.81,3.86,3.92,3.97,4.02,4.08,4.13,4.17,4.22,4.27,4.31,4.36,4.41,4.44,4.49,4.52,4.56,4.60,4.64,4.67,4.71,4.74,4.77,4.80,4.82,4.85,4.87,4.89,4.91,4.93,4.94,4.96,4.97,4.98,4.99,4.99,4.99,4.99,4.99,4.99,4.98,4.97,4.96,4.94,4.93,4.91,4.88,4.86,4.83,4.80,4.77,4.73,4.70,4.66,4.62,4.57,4.52,4.46,4.42,4.36,4.29,4.24,4.18,4.11,4.06,3.99,3.92,3.85,3.78,3.70,3.63,3.55,3.48,3.41,3.33,3.26,3.18,3.09,3.02,2.94,2.85,2.78,2.69,2.61,2.54,2.45,2.37,2.30,2.21,2.13,2.06,1.98,1.89,1.82,1.74,1.67,1.59,1.52,1.45,1.37,1.30,1.23,1.16,1.09,1.03,0.96,0.90,0.84,0.78,0.72,0.67,0.61,0.55,0.51,0.45,0.41,0.36,0.32,0.28,0.24,0.21,0.18,0.14,0.12,0.09,0.07,0.05,0.04,0.02,0.01,0.01,0.00]
    x = [1, 0, 0, 1]
    y = [0, 0, 1, 1]
    # y=[37.23,38.91,40.61,41.66,43.01,45.78,49.20,51.85,53.81,56.15,58.65,57.61,55.97,54.22,52.13,50.91,51.01,51.65,52.28,53.65,54.56,54.53,54.43,53.75,52.45,51.85,51.76,51.75,51.80,52.42,52.42,52.47,52.60,52.75,52.83,52.55,52.35,52.25,52.01,51.82,51.82,51.81,51.85,51.88,51.88,51.81,51.80,51.75,51.53,51.49,51.54,51.51,51.51,51.52,51.51,51.48,51.52,51.26,51.09,51.05,50.92,50.93,50.97,50.97,50.95,51.02,50.99,51.04,51.04,50.92,50.65,50.64,50.61,50.61,50.66,50.67,50.64,50.67,50.58,50.47,50.45,50.24,50.07,50.10,50.07,50.05,50.11,50.10,50.07,49.97,49.70,49.67,49.68,49.50,49.50,49.49,49.47,49.50,49.46,49.48,49.21,48.11,47.81,47.37,47.32,46.85,45.77,44.54,43.09,41.66,40.29,38.49,36.54,33.99,31.23,28.23,25.26,23.25,24.20,26.10,29.01,31.74,33.24,33.20,32.61,30.41,27.65,26.16,25.95,25.98,27.61,29.39,31.12,31.89,31.97,30.75,29.65,28.33,27.31,27.00,27.47,28.33,29.30,30.26,30.96,30.99,30.31,29.17,28.83,28.18,28.16,28.18,28.94,29.49,30.08,30.34,30.43,30.24,29.58,29.15,29.08,29.08,29.41,29.76,30.36,30.48,30.55,30.48,30.47,30.14,29.80,29.80,30.17,30.39,30.85,31.42,31.55,31.53,31.54,31.48,31.43,31.40,31.41,31.57,32.01,32.66,33.24,33.25,33.24,33.24,32.80,32.25,32.25,32.40,32.61,33.04]
    for index in range(len(x)):
        points.append(Point(x[index], y[index]))

    area = GetAreaOfPolyGon(points)
    print(area)