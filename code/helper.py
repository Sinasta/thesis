import zipfile
import os
import tempfile
import pathlib
import json
from tqdm.auto import tqdm

import topologicpy
import topologic
import multiprocessing

import torch
from dgl import save_graphs, load_graphs
from topologicpy.Topology import Topology
from topologicpy.DGL import DGL
from topologicpy.Graph import Graph
from topologicpy.Dictionary import Dictionary

def energy_to_class(energy_consumption, energy_quantiles):
    return next((i for i, q in enumerate(energy_quantiles) if energy_consumption <= q), len(energy_quantiles))

def categories():
    with open('./graph/conversion_list.json', 'r') as cl:
        conversion_list = json.load(cl)
    return list(range(len(conversion_list)))

def save_graph_batch(ccomplex_database, name, categories):
    dgl_graphs = []
    labels = []
    values = []
    for ccomplex_variant_list in ccomplex_database.values():
            for ccomplex_tuple in ccomplex_variant_list:
                consumption, graph, energy_class = ccomplex_tuple
                dgl_graph = DGL.ByGraph(graph, bidirectional=True, key='label', categories=categories, node_attr_key='node_attr', tolerance=0.0001)
                dgl_graphs.append(dgl_graph)
                labels.append(energy_class)
                values.append(consumption)
    graph_path = './graph/data/graph_batches_new/graph_batch_' + name + '.bin'
    graph_labels = {'label': torch.tensor(labels), 'value': torch.tensor(values)}
    save_graphs(graph_path, dgl_graphs, graph_labels)

def rebuilding(frame, categories, quantiles_dict):
    path = './graph/data/geometry'
    with zipfile.ZipFile(os.path.join(path, 'geometry_batch_' + str(frame) + '.zip')) as frame_zip:
        cellcomplex_variants = {}
        for rotation in range(4):
            ccomplex_list = []
            for variant in range(4):
                json_file = frame_zip.open(os.path.join(str(rotation), (str(variant) + '.json')))
                data = json.load(json_file)
                temp_dir = tempfile.TemporaryDirectory()
                tmp_json = os.path.join(temp_dir.name, 'tmp.json')
                with open(tmp_json, 'w') as tmp_apartment:
                    json.dump(data, tmp_apartment, indent=4)
                tmp_apartment.close()
                restart = True
                count = 0
                while restart and (count < 20):
                    try:
                        apartment = Topology.ByImportedJSONMK1(tmp_json)
                        graph = Graph.ByTopology(apartment[0], toExteriorApertures=True, useInternalVertex=False)
                        restart = False
                    except:
                        restart = False
                        count += 1
                ccdict = Dictionary.PythonDictionary(Topology.Dictionary(apartment[0]))
                consumption = ccdict['total_site_energy_consumption_per_surface_MJ/m2']
                energy_class = energy_to_class(consumption, quantiles_dict['energy'])
                ccomplex_list.append((consumption, graph, energy_class))
                temp_dir.cleanup()
            cellcomplex_variants[rotation] = ccomplex_list
        save_graph_batch(cellcomplex_variants, str(frame), categories)
        frame_zip.close()

if __name__ == "__main__":
    with open('./graph/quantiles_dict.json', 'r') as qd:
        quantiles_dict = json.load(qd)
    categories = categories()
    amount = 2500
    batch_size = multiprocessing.cpu_count()
    for i in tqdm(range(0, amount, batch_size), desc='rebuilding'):
        processes = [multiprocessing.Process(target=rebuilding, args=(frame, categories, quantiles_dict)) for frame in range(i, i+batch_size) if frame < amount]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
