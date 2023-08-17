from generation.tools import *
from generation.outline import *
from generation.algorithms.treemap import *
from generation.algorithms.voronoi_treemap import *
from generation.algorithms.voronoi import *
from energy.add_apertures_energy_sim import *
from graph.tools import *

import multiprocessing
import os

def generation(frame, seed):
    geometry_file = './graph/data/geometry/geometry_batch_' + str(frame) + '.zip'
    graph_file = './graph/data/graph_batches/graph_batch_' + str(frame) + '.bin'
    if os.path.exists(graph_file) and os.path.exists(geometry_file):
        return
    if seed:
        seed += 100 * (frame + 1)
    else:
        seed = np.random.randint(9**9)
    done = False
    while not done:
        try:
            outline = pick_outline_method(seed=seed)
            room_amount = room_amount_from_total_area(outline)
            room_sizes = create_rooms(room_amount, seed=seed)
            names_sorted = get_sorted_names(room_sizes)
            percentages_list = normalize(room_sizes)
            ratio_tree_dict = build_tree_list(percentages_list, seed=seed)
            layout, layout_type = pick_layout_method(seed, outline, ratio_tree_dict)
            layout = add_room_names(layout, names_sorted)
            done = check_if_layout_good(layout)
            if not done:
                seed += 1
                continue
            ccomplex_database = add_apertures_energy_sim(layout)
        except RuntimeError:
            done = False
            seed += 1
    save_geometry_batch(ccomplex_database, str(frame))
    save_graph_batch(ccomplex_database, str(frame), num_nlabels=91)

def main(seed=1):
    create_directories()
    if int(input('[0] classification\n[1] regression\ntask: ')):
        task = 'regression'
    else:
        task = 'classification'
    amount = int(input('amount: '))
    batch_size = multiprocessing.cpu_count()
    for i in tqdm(range(0, amount, batch_size), desc='Generating'):
        processes = [multiprocessing.Process(target=generation, args=(frame, seed)) for frame in range(i, i+batch_size) if frame < amount]
        for process in processes:
            process.start()
        for process in processes:
            process.join()

    if task.lower() == 'regression':
        cv_type = 'K-Fold'
        datasets = build_dataset(amount, fracList=[0.8, 0.1, 0.1], shuffle=False, randomState=1, label_key='value')
        results, dataloaders = train(datasets, batch_size=64, epochs=350, lr=0.1, hl_widths=[32], conv_layer_type='TagConv', cv_type=cv_type, k_folds=10, model_type='Regressor')
    elif task.lower() == 'classification':
        cv_type = 'K-Fold'
        datasets = build_dataset(amount, fracList=[0.8, 0.1, 0.1], shuffle=False, randomState=1, label_key='label')
        results, dataloaders = train(datasets, batch_size=64, epochs=50, lr=0.01, hl_widths=[16,16], conv_layer_type='GraphConv', cv_type=cv_type, k_folds=10, loss_function='negative log likelihood', pooling='AvgPooling', model_type='Classifier')
    
    save_figs(results, dataloaders, datasets, format_type='svg', export=True, task=task, cv_type=cv_type)

if __name__ == "__main__":
    main()
