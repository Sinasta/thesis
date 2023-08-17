import topologicpy
import topologic

from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Wire import Wire
from topologicpy.Face import Face
from topologicpy.Shell import Shell
from topologicpy.Cluster import Cluster
from topologicpy.Topology import Topology
from topologicpy.Dictionary import Dictionary
import pyvoro

from scipy.optimize import fmin
from scipy.spatial import Voronoi
import numpy as np

def generate_points(polygon:topologic.Face, amount:int, seed:int) -> topologic.Cluster:
    np.random.seed(seed)
    points = []
    bounds = Face.Vertices(Face.BoundingRectangle(polygon))
    minx, miny = Vertex.Coordinates(bounds[0], outputType='xy')
    maxx, maxy = Vertex.Coordinates(bounds[2], outputType='xy')
    
    while len(points) < amount:
        rand_coord = (np.random.uniform(minx, maxx), np.random.uniform(miny, maxy), 0)
        pnt = Vertex.ByCoordinates(*rand_coord)
        if Vertex.IsInside(pnt, polygon):
            if len(points) > 0:
                distances = []
                for vertex in points:
                    distance = Vertex.Distance(pnt, vertex)
                    distances.append(distance)
                shortest_distance = sorted(distances)[0]
                if shortest_distance >= 0.5:
                    points.append(pnt)
            else:
                points.append(pnt)
    point_cluster = Cluster.ByTopologies(points)
    return point_cluster
    
def lloyd_iteration(points:topologic.Cluster, polygon:topologic.Face, seed:int) -> topologic.Cluster:
    np.random.seed(seed)
    iterations = np.random.randint(5)
    point_list = Cluster.Vertices(points)
    if iterations == 0:
        return points
    for iteration in range(iterations):
        voronoi = Shell.Voronoi(point_list, polygon)
        cells = Topology.SubTopologies(voronoi, subTopologyType='face')
        centers = []
        for cell in cells:
            center = Topology.Centroid(cell)
            centers.append(center)
        point_list = centers
    point_cluster = Cluster.ByTopologies(point_list)
    return point_cluster
    
def create_voronoi_two_points(polygon:topologic.Face, radius:list, points:topologic.Cluster, start_value):
    point_list = Cluster.Vertices(points)
    dots = []
    for point in point_list:
        dot = Vertex.Coordinates(point, outputType='xy')
        dots.append(dot)
    radii_array = np.array([start_value, radius[0]])
    bounds = Face.Vertices(Face.BoundingRectangle(polygon))
    minx, miny = Vertex.Coordinates(bounds[0], outputType='xy')
    maxx, maxy = Vertex.Coordinates(bounds[2], outputType='xy')
    try:
        voro_cells = pyvoro.compute_2d_voronoi(dots, [[minx-1, maxx+1], [miny-1, maxy+1]], 1, radii=radii_array)
        regions = []
        for voro_cell in voro_cells:
            vertices = []
            voro_verts = voro_cell['vertices']
            for vert in voro_verts:
                vertex = Vertex.ByCoordinates(*vert, 0)
                vertices.append(vertex)
            wire = Wire.ByVertices(vertices, close=True)
            region = Face.TrimByWire(polygon, wire, reverse=True)
            regions.append(region)
    except:
        region_1 = Face.Rectangle(width=0.01, length=0.01)
        region_2 = polygon
        regions = [region_1, region_2]
    return regions
    
def find_weights(polygon, ratio, seed_delta, seed):
    total_area = Face.Area(polygon)
    points = lloyd_iteration(generate_points(polygon, 2, seed_delta), polygon, seed)
    def change_weight(radius, polygon, ratio, total_area, points, start_value):
        regions = create_voronoi_two_points(polygon, radius, points, start_value)
        area = Face.Area(regions[0])
        if not area:
            area = 0
        diff = np.abs(total_area * ratio - area)
        return diff
    box = Face.BoundingRectangle(polygon, optimize=4)
    edges = Face.Edges(box)
    lengths = []
    for edge in edges:
        length = Edge.Length(edge)
        lengths.append(length)
    start_value = min(lengths)

    radius, diff, *_ = fmin(change_weight, start_value, maxiter=10, full_output=True, disp=0, args=(polygon, ratio, total_area, points, start_value))
    return radius, points, start_value
    
def recursive_divison_voronoi(polygon, ratio_tree_dict, seed):
    list_of_polygons = [polygon]
    for level in range(1, len(ratio_tree_dict)):
        list_of_child_polygons = []
        level_name = 'level_' + str(level)
        for leave in range(len(ratio_tree_dict[level_name])):
            if ratio_tree_dict[level_name][leave][0] != 0.0:
                ratio = ratio_tree_dict[level_name][leave][0]
                polygon = list_of_polygons[leave]
                seed_delta = np.random.randint(1000)
                radius, points, start_value = find_weights(polygon, ratio, seed_delta, seed)
                two_child_polygons = create_voronoi_two_points(polygon, radius, points, start_value)
                list_of_child_polygons.extend(two_child_polygons)
            else:
                list_of_child_polygons.append(list_of_polygons[leave])
        list_of_polygons = list_of_child_polygons
        voronoi_shell = Shell.ByFaces(list_of_polygons)
    return voronoi_shell
