import topologicpy
import topologic

from topologicpy.Edge import Edge
from topologicpy.Face import Face
from topologicpy.Shell import Shell
from topologicpy.Topology import Topology
from topologicpy.Plotly import Plotly
from topologicpy.Dictionary import Dictionary

from generation.algorithms.treemap import *
from generation.algorithms.voronoi_treemap import *
from generation.algorithms.voronoi import *

import numpy as np
import json

def room_amount_from_total_area(polygon:topologic.Face) -> int:
    area = Face.Area(polygon)
    room_amount = None
    boundary = [27.5,47.5,54,66,76.5,83,96,115.5,136.5]
    room_amounts = range(3,11)
    room_amount = np.digitize(area, boundary, right=True)
    if room_amount == 0 or room_amount > len(room_amounts):
        raise Exception('Surface Area is too small')
    elif room_amount > len(room_amounts):
        raise Exception('Surface Area is too big')
    return room_amounts[room_amount -1]
    
def create_rooms(room_amount:int, seed=None) -> dict:
    room_dict = {}
    if seed:
        np.random.seed(seed)
    with open('./generation/room_specs.json', 'r') as room_specs_json:
        room_specs = json.load(room_specs_json)
    for room_name, min_max_area in room_specs[str(room_amount)].items():
        min_area = min_max_area[0]
        max_area = min_max_area[1]
        room_dict[room_name] = round(np.random.uniform(min_area, max_area), 2)
    return room_dict
    
def get_sorted_names(room_sizes_dict:dict) -> list:
    sorted_room_sizes_dict = sorted(room_sizes_dict.items(), key=lambda x: x[1])
    sorted_room_names = [x[0] for x in sorted_room_sizes_dict]
    return sorted_room_names
    
def normalize(sizes_dict:dict) -> list:
    values = np.array(list(sizes_dict.values()))
    total = values.sum()
    return values/total
    
def build_tree_list(percentages_list: list, seed:int)  -> dict:
    np.random.seed(seed)
    np.random.shuffle(percentages_list)
    tree_depth = int(np.ceil(np.log2(len(percentages_list))))
    parent_list = [percentages_list]
    tree_sum_dict = {'level_0': [[1]]}
    for level in range(tree_depth):
        new_parent_list = []
        new_sum_list = []
        for leave in range(2**(level)):
            if len(parent_list[leave]) > 1:
                split_index = int(len(parent_list[leave]) / 2)
                if len(parent_list[leave]) % 2 != 0:
                    left_data, right_data = np.split(parent_list[leave], [split_index])
                    if np.sum(left_data) < np.sum(right_data):
                        split_index += 1
                first_child, second_child = np.split(parent_list[leave], [split_index])
            else:
                first_child = np.array([0])
                second_child = np.array([0])
            new_sum_list.append([np.sum(first_child), np.sum(second_child)])
            new_parent_list.extend([first_child, second_child])
        level_name = 'level_' + str(level + 1)
        tree_sum_dict[level_name] = new_sum_list
        parent_list = new_parent_list
    ratio_tree_dict = {}
    ratio_tree_dict = {'level_0': [[1]]}
    for level in range(1, len(tree_sum_dict)):
        child_list = []
        level_name = 'level_' + str(level)
        for index in range(len(tree_sum_dict[level_name])):
            factor = 1 / tree_sum_dict['level_' + str(level-1)][int(index/2)][index%2]
            new_first = tree_sum_dict[level_name][index][0] * factor
            new_second = tree_sum_dict[level_name][index][1] * factor
            child_list.append([new_first, new_second])
        ratio_tree_dict[level_name] = child_list
    return ratio_tree_dict
    
def check_if_layout_good(shell):
    face_list = Shell.Faces(shell)
    for face in face_list:
        
        face_edges = Face.Edges(face)
        edges_lengths = []
        for edge in face_edges:
            edges_lengths.append(Edge.Length(edge))
        shortest_wall = sorted(edges_lengths)[0]
        
        box = Face.BoundingRectangle(face, optimize=10)
        edges = Face.Edges(box)
        min_length, max_length = sorted([Edge.Length(edges[0]), Edge.Length(edges[1])])
        
        ratio = min_length / max_length
        
        area = Face.Area(face)
        
        compactness = Face.Compactness(face)
        
        if min_length <= 1.5 and area > 4:
            return False
        elif min_length <= 0.7 and area <= 4:
            return False
        elif ratio <= 0.3 and area > 4:
            return False
        elif ratio <= 0.4 and area <= 4:
            return False
        elif min_length <= 0.1:
            return False
        elif area <= 0.5:
            return False
        elif shortest_wall <= 0.3:
            return False
        else:
            return True
    
def add_room_names(shell, names_sorted):
    face_list = Shell.Faces(shell)
    area_name_dict = {}
    for face in face_list:
        area_name_dict[face] = Face.Area(face)
    sorted_faces = sorted(area_name_dict, key=area_name_dict.get)
    labeled_faces = []
    for index, face in enumerate(sorted_faces):
        pythonDictionary = {'type' : names_sorted[index]}
        dictionary = Dictionary.ByPythonDictionary(pythonDictionary)
        labeled_face = Topology.AddDictionary(face, dictionary)
        labeled_faces.append(labeled_face)
    labeled_shell = Shell.ByFaces(labeled_faces)
    return labeled_shell
    
def pick_layout_method(seed:int, outline, ratio_tree_dict) -> tuple:
    np.random.seed(seed)
    pick = np.random.randint(3)
    if pick == 0:
        return recursive_divison_treemap(outline, ratio_tree_dict), 'treemap'
    elif pick == 1:
        treemap = recursive_divison_treemap(outline, ratio_tree_dict)
        return treemap_to_voronoi(*find_radius_voronoi(outline, ratio_tree_dict, treemap)), 'voronoi_treemap'
    elif pick == 2:
        return recursive_divison_voronoi(outline, ratio_tree_dict, seed), 'voronoi'
