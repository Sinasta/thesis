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

def find_poly_depth(polygon:topologic.Face) -> int:
    edges = Face.Edges(Face.BoundingRectangle(polygon))
    if Edge.Length(edges[0]) >= Edge.Length(edges[1]):
        direction = 'x'
    else:
        direction = 'y'
    return direction
    
def create_cutline(position:float, direction:str) -> topologic.Face:
    if direction == 'y':
        point = (0, position, 0)
        dirX = 0
        dirY = 1
    else:
        point = (position, 0, 0)
        dirX = 1
        dirY = 0
    origin = Vertex.ByCoordinates(*point)
    section_plane = Face.Rectangle(origin=origin, width=1.0, length=30, direction=[dirX, dirY, 0])
    return section_plane
    
def split_polygon(polygon:topologic.Face, section_plane:topologic.Face) -> list:
    split_cluster = Topology.Boolean(polygon, section_plane, operation='difference')
    split_polygons = Topology.SubTopologies(split_cluster, subTopologyType='face')
    return split_polygons
    
def find_cut_position(polygon:topologic.Face, ratio:float, direction:str) -> float:
    total_area = Face.Area(polygon)
    origin = Topology.Centroid(polygon)

    def cut_polygon(delta, polygon, ratio, total_area, direction, origin):
        center = Vertex.Coordinates(origin, outputType=direction)[0]
        section_plane = create_cutline(center + delta, direction)
        split_polygons = split_polygon(polygon, section_plane)
        area = Face.Area(split_polygons[0])
        diff = np.abs(total_area * ratio - area)
        return diff
    cut_position, diff, *_ = fmin(cut_polygon, 0, maxiter=100, full_output=True, disp=0, args=(polygon, ratio, total_area, direction, origin))

    center = Vertex.Coordinates(origin, outputType=direction)[0]
    abs_cut_position = center + cut_position[0]
    return abs_cut_position
    
def recursive_divison_treemap(polygon:topologic.Face, ratio_tree_dict:dict) -> topologic.Shell:
    list_of_polygons = [polygon]
    for level in range(1, len(ratio_tree_dict)):
        list_of_child_polygons = []
        level_name = 'level_' + str(level)
        for leave in range(len(ratio_tree_dict[level_name])):
            if ratio_tree_dict[level_name][leave][0] != 0.0:
                ratio = ratio_tree_dict[level_name][leave][0]
                polygon = list_of_polygons[leave]
                
                direction = find_poly_depth(polygon)
                
                position = find_cut_position(polygon, ratio, direction)
                
                section_plane = create_cutline(position, direction)
                two_child_polygons = split_polygon(polygon, section_plane)
                list_of_child_polygons.extend(two_child_polygons)
                
            else:
                list_of_child_polygons.append(list_of_polygons[leave])
        list_of_polygons = list_of_child_polygons
        treemap_shell = Shell.ByFaces(list_of_polygons)
    return treemap_shell
