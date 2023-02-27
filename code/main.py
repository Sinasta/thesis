from generation.tools import *
from generation.outline import *
from generation.algorithms.treemap import *
from generation.algorithms.voronoi_treemap import *
from generation.algorithms.voronoi import *

from energy.add_apertures_energy_sim import *

import topologicpy
import topologic

import json
from topologicpy.Graph import Graph
from topologicpy.Topology import Topology
from topologicpy.Dictionary import Dictionary
from topologicpy.DGL import DGL
from tqdm import tqdm

def main(amount = 40, seed = None):
    ccomplex_database_list = []
    for i in tqdm(range(1, amount + 1)):
        if seed:
            seed += 1 + i
        else:
            seed = np.random.randint(9**9)
        outline = pick_outline_method(seed=seed)
        room_amount = room_amount_from_total_area(outline)
        room_sizes = create_rooms(room_amount, seed=seed)
        names_sorted = get_sorted_names(room_sizes)
        percentages_list = normalize(room_sizes)
        ratio_tree_dict = build_tree_list(percentages_list, seed=seed)
        layout, layout_type = pick_layout_method(seed, outline, ratio_tree_dict)
        layout = add_room_names(layout, names_sorted)
        check_if_layout_good(layout)
        
        ccomplex_database = add_apertures_energy_sim(layout)
        ccomplex_database_list.append(ccomplex_database)
    
    return ccomplex_database_list

if __name__ == "__main__":
    ccomplex_database_list = main()
    def energy_to_class(energy_consumption, energy_quantiles):
        return next((i+1 for i, q in enumerate(energy_quantiles) if energy_consumption <= q), len(energy_quantiles)+1)
    def categories():
        with open('./graph/conversion_list.json', 'r') as cl:
            conversion_list = json.load(cl)
        return list(range(len(conversion_list)))
        
    with open('./graph/quantiles_dict.json', 'r') as qd:
        quantiles_dict = json.load(qd)
    
    dgl_graphs = []
    graph_labels = []
    for ccomplex_database in ccomplex_database_list:
        for ccomplex_variant_list in ccomplex_database.values():
            for ccomplex in ccomplex_variant_list:
                ccdict = Dictionary.PythonDictionary(Topology.Dictionary(ccomplex))
                consumption = ccdict['total_site_energy_consumption_per_surface_MJ/m2']
                energy_class = energy_to_class(consumption, quantiles_dict['energy'])
                graph_labels.append(energy_class)
                graph = Graph.ByTopology(ccomplex, toExteriorApertures=True, useInternalVertex=False)
                dgl_graph = DGL.ByGraph(graph, bidirectional=True, key='label', categories=categories(), node_attr_key='node_attr', tolerance=0.0001)
                dgl_graphs.append(dgl_graph)
    dataset = DGL.DatasetByDGLGraphs(dgl_graphs, graph_labels, key='node_attr')
    
    hyper_param = DGL.Hyperparameters(
        optimizer=DGL.Optimizer(), 
        hl_widths=[32],
        conv_layer_type='GraphConv',
        batch_size=1, 
        epochs=10,
        loss_function='Cross Entropy',
        classifier_path='./graph', 
        results_path='./graph')
    results = DGL.Train(hyper_param, dataset)
    print(results)
