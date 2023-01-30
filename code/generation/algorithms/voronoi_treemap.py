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

def create_voronoi(polygon, radius, points):
    point_list = Cluster.Vertices(points)
    dots = []
    for point in point_list:
        dot = Vertex.Coordinates(point, outputType='xy')
        dots.append(dot)
    radii_array = np.array(radius)
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
            region = Face.ByVertices(vertices)
            regions.append(region)
        outside_voronoi_shell = Shell.ByFaces(regions)
        final_regions = Topology.Boolean(polygon, outside_voronoi_shell, operation='slice')
        regions = Shell.Faces(final_regions)
    except:
        empty_regions = []
        for i in range(len(radius)):
            empty_regions.append(Face.Rectangle(width=0.1, length=0.1))
        regions = Shell.Faces(empty_regions)
    return regions
    
def treemap_to_voronoi(factor, outline, treemap):
    points = []
    radius = []
    total = Face.Area(outline)
    treemap_faces = Shell.Faces(treemap)
    for i, region in enumerate(treemap_faces):
        points.append(Topology.Centroid(region))
        percent = Face.Area(region) / total * factor
        radius.append(percent)
    points = Cluster.ByTopologies(points)
    voronoi_cells = create_voronoi(outline, radius, points)
    treemap_voronoi_shell = Shell.ByFaces(voronoi_cells)
    return treemap_voronoi_shell
    
def find_radius_voronoi(outline, ratio_tree_dict, treemap):
    treemap_area = []
    treemap_faces = Shell.Faces(treemap)
    for region in treemap_faces:
        treemap_area.append(Face.Area(region))
    treemap_array = np.sort(np.array(treemap_area))
    def change_radius(factor, outline, treemap, treemap_array):
        voronoi_area = []
        tree_voronoi = treemap_to_voronoi(factor[0], outline, treemap)
        try:
            for cell in Shell.Faces(tree_voronoi):
                voronoi_area.append(Face.Area(cell))
            voronoi_array = np.sort(np.array(voronoi_area))
            if len(voronoi_array) != len(treemap_array):
                voronoi_array = np.zeros(len(treemap_array))
            diff = np.sum(np.abs(treemap_array - voronoi_array))
        except:
            diff = 100
        return diff
    start_value = 5
    optimized_factor, diff, *_ = fmin(change_radius, start_value, maxiter=100, full_output=True, disp=0, args=(outline, treemap, treemap_array))
    return optimized_factor[0], outline, treemap
