import topologicpy
import topologic

from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Face import Face

import numpy as np

def generate_outline_polygon(seed:int = None):
    area = 0
    ratio = 0
    while (area < 27.5 or area > 136.5) or ratio < 0.7:
        np.random.seed(seed)
        n = np.random.randint(3, 7)
        if n == 3:
            scale = 32.5
        elif n == 4:
            scale = 22.6
        elif n == 5:
            scale = 18.8
        elif n == 6:
            scale = 16.7

        X_rand, Y_rand = np.sort(np.random.random(n)) * scale, np.sort(np.random.random(n)) * scale
        X_new, Y_new = np.zeros(n), np.zeros(n)

        last_true = last_false = 0
        for i in range(1, n):
            if i != n - 1:
                if np.random.randint(2):
                    X_new[i] = X_rand[i] - X_rand[last_true]
                    Y_new[i] = Y_rand[i] - Y_rand[last_true]
                    last_true = i
                else:
                    X_new[i] = X_rand[last_false] - X_rand[i]
                    Y_new[i] = Y_rand[last_false] - Y_rand[i]
                    last_false = i
            else:
                X_new[0] = X_rand[i] - X_rand[last_true]
                Y_new[0] = Y_rand[i] - Y_rand[last_true]
                X_new[i] = X_rand[last_false] - X_rand[i]
                Y_new[i] = Y_rand[last_false] - Y_rand[i]

        np.random.shuffle(Y_new)
        vertices = np.stack((X_new, Y_new), axis=-1)
        vertices = vertices[np.argsort(np.arctan2(vertices[:, 1], vertices[:, 0]))]

        vertices = np.cumsum(vertices, axis=0)

        x_max, y_max = np.max(vertices[:, 0]), np.max(vertices[:, 1])
        vertices[:, 0] += ((x_max - np.min(vertices[:, 0])) / 2) - x_max
        vertices[:, 1] += ((y_max - np.min(vertices[:, 1])) / 2) - y_max

        points = []
        for coord in vertices:
            point = Vertex.ByCoordinates(coord[0], coord[1], 0)
            points.append(point)
        polygon = Face.ByVertices(points)
        
        area = Face.Area(polygon)
        box = Face.BoundingRectangle(polygon, optimize=4)
        edges = Face.Edges(box)
        sizes = [Edge.Length(edges[0]), Edge.Length(edges[1])]
        ratio = min(sizes) / max(sizes)
        if seed:
            seed += np.random.randint(9**5)
    return polygon
    
def generate_outline_rectangle(seed:int = None) -> topologic.Face:
    area = ratio = 0
    while (area < 27.5 or area > 136.5) or ratio < 0.7:
        np.random.seed(seed)
        sizes = np.random.uniform(4, 19, 2).tolist()
        rectangle = Face.Rectangle(width=sizes[0], length=sizes[1])
        area = Face.Area(rectangle)
        ratio = min(sizes) / max(sizes)
        if seed:
            seed += np.random.randint(9**5)
    return rectangle
    
def generate_outline_square(seed:int = None) -> topologic.Face:
    np.random.seed(seed)
    size = np.random.uniform(5.245, 11.683)
    return Face.Rectangle(width=size, length=size)
    
def pick_outline_method(seed:int = None) -> topologic.Face:
    np.random.seed(seed)
    methods = [generate_outline_square(seed), generate_outline_rectangle(seed), generate_outline_polygon(seed)]
    return np.random.choice(methods, 1, p=[0.2, 0.4, 0.4])[0]
