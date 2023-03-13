from generation.tools import *
from generation.outline import *
from generation.algorithms.treemap import *
from generation.algorithms.voronoi_treemap import *
from generation.algorithms.voronoi import *
from energy.add_apertures_energy_sim import *
from graph.tools import *

import os

def main(seed=1):
    create_directories()
    cache = input('cache y/n: ')
    if cache.lower() == 'y':
        dataset, validation_set = load_csv_dataset(validation_split=0.2)
    else:
        amount = int(input('amount: '))
        for frame in tqdm(range(amount), desc='computing graphs'):
            if os.path.exists('./graph/data/graph_batches/graph_batch_' + str(frame) + '.bin'):
                continue
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
            save_graph_batch(ccomplex_database, str(frame))
            save_geometry_batch(ccomplex_database, str(frame))
        dataset, validation_set = build_dataset(amount, validation_split=0.2)
    results = train(dataset, validationDataset=validation_set, batch_size=100, epochs=200, lr=0.001, hl_widths=[32], conv_layer_type='TAGConv', cv_type='Holdout', loss_function='Negative Log Likelihood', optimizer='Adam')
    save_figs(results, dataset, validation_set, format_type='svg', export=True)
    return results, dataset

if __name__ == "__main__":
    main()
