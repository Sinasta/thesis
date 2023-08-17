import topologicpy
import topologic

from topologicpy.Graph import Graph
from topologicpy.EnergyModel import EnergyModel
from topologicpy.Topology import Topology
from topologicpy.Dictionary import Dictionary
from topologicpy.CellComplex import CellComplex
from topologicpy.Cell import Cell
from topologicpy.Face import Face

import numpy as np
import os
import sys
import zipfile
import pathlib
import json
import tempfile

def energy_to_class(energy_consumption, energy_quantiles):
    return next((i for i, q in enumerate(energy_quantiles) if energy_consumption <= q), len(energy_quantiles))

def energy_simulation(Apartment, name):
    directory = os.getcwd()
    osModelPath = directory + '/energy/midrise-apartment.osm'
    weatherFilePath = directory + '/energy/Berlin.epw'
    designDayFilePath = directory + '/energy/Berlin.ddy'
    buildingTopology = Apartment
    shadingSurfaces = None
    floorLevels = [0.0, 2.7]
    buildingName = 'Apartment_' + name
    buildingType = 'Residential'
    defaultSpaceType = 'Apartment'
    northAxis = 0.00
    glazingRatio = 0.00
    coolingTemp = 25.00
    heatingTemp = 20.00
    energy_model = EnergyModel.ByTopology(
        buildingTopology, shadingSurfaces, osModelPath, weatherFilePath,
        designDayFilePath, floorLevels, buildingName, buildingType, northAxis,
        glazingRatio, coolingTemp, heatingTemp, defaultSpaceType, 'type', 'ep_type')
    tmp = tempfile.TemporaryDirectory()
    sim_model = EnergyModel.Run(energy_model, weatherFilePath, directory + '/energy/openstudio/bin/openstudio', tmp.name)
    EPReportName = [
        'AnnualBuildingUtilityPerformanceSummary',
        'AnnualBuildingUtilityPerformanceSummary',
        'AnnualBuildingUtilityPerformanceSummary',
        'AnnualBuildingUtilityPerformanceSummary',
        'AnnualBuildingUtilityPerformanceSummary',
        'AnnualBuildingUtilityPerformanceSummary',
        'AnnualBuildingUtilityPerformanceSummary',
        'AnnualBuildingUtilityPerformanceSummary',
        'ObjectCountSummary',
        'ObjectCountSummary',
        'InputVerificationandResultsSummary',
        'InputVerificationandResultsSummary',
        'InputVerificationandResultsSummary',
        'InputVerificationandResultsSummary',
        'InputVerificationandResultsSummary',
        'InputVerificationandResultsSummary',
        'InputVerificationandResultsSummary',
        'InputVerificationandResultsSummary',
        'InputVerificationandResultsSummary',
        'InputVerificationandResultsSummary',
        'InputVerificationandResultsSummary',
        'InputVerificationandResultsSummary',
        'InputVerificationandResultsSummary',
        'InputVerificationandResultsSummary',
        'InputVerificationandResultsSummary',
    ]   
    EPReportForString = [
        'Entire Facility',
        'Entire Facility',
        'Entire Facility',
        'Entire Facility',
        'Entire Facility',
        'Entire Facility',
        'Entire Facility',
        'Entire Facility',
        'Entire Facility',
        'Entire Facility',
        'Entire Facility',
        'Entire Facility',
        'Entire Facility',
        'Entire Facility',
        'Entire Facility',
        'Entire Facility',
        'Entire Facility',
        'Entire Facility',
        'Entire Facility',
        'Entire Facility',
        'Entire Facility',
        'Entire Facility',
        'Entire Facility',
        'Entire Facility',
        'Entire Facility',
    ]
    EPTableName = [
        'Site and Source Energy',
        'Site and Source Energy',
        'Site and Source Energy',
        'Site and Source Energy',
        'End Uses',
        'End Uses',
        'Utility Use Per Total Floor Area',
        'Utility Use Per Total Floor Area',
        'Surfaces by Class',
        'Surfaces by Class',
        'Window-Wall Ratio',
        'Window-Wall Ratio',
        'Window-Wall Ratio',
        'Window-Wall Ratio',
        'Window-Wall Ratio',
        'Window-Wall Ratio',
        'Window-Wall Ratio',
        'Window-Wall Ratio',
        'Window-Wall Ratio',
        'Window-Wall Ratio',
        'Window-Wall Ratio',
        'Window-Wall Ratio',
        'Window-Wall Ratio',
        'Window-Wall Ratio',
        'Window-Wall Ratio',
    ]
    EPRowName = [
        'Total Source Energy',
        'Total Site Energy',
        'Total Source Energy',
        'Total Site Energy',
        'Heating',
        'Cooling',
        'Total',
        'Total',
        'Wall',
        'Window',
        'Gross Wall Area',
        'Gross Wall Area',
        'Gross Wall Area',
        'Gross Wall Area',
        'Gross Wall Area',
        'Window Opening Area',
        'Window Opening Area',
        'Window Opening Area',
        'Window Opening Area',
        'Window Opening Area',
        'Gross Window-Wall Ratio',
        'Gross Window-Wall Ratio',
        'Gross Window-Wall Ratio',
        'Gross Window-Wall Ratio',
        'Gross Window-Wall Ratio'
    ]
    EPColumnName = [
        'Total Energy',
        'Total Energy',
        'Energy Per Total Building Area',
        'Energy Per Total Building Area',
        'District Heating',
        'District Cooling',
        'District Heating Intensity',
        'District Cooling Intensity',
        'Outdoors',
        'Total',
        'Total',
        'North (315 to 45 deg)',
        'East (45 to 135 deg)',
        'South (135 to 225 deg)',
        'West (225 to 315 deg)',
        'Total',
        'North (315 to 45 deg)',
        'East (45 to 135 deg)',
        'South (135 to 225 deg)',
        'West (225 to 315 deg)',
        'Total',
        'North (315 to 45 deg)',
        'East (45 to 135 deg)',
        'South (135 to 225 deg)',
        'West (225 to 315 deg)'
    ]
    EPUnits = [
        'GJ',
        'GJ',
        'MJ/m2',
        'MJ/m2',
        'GJ',
        'GJ',
        'MJ/m2',
        'MJ/m2',
        '',
        '',
        'm2',
        'm2',
        'm2',
        'm2',
        'm2',
        'm2',
        'm2',
        'm2',
        'm2',
        'm2',
        '%',
        '%',
        '%',
        '%',
        '%'
    ]
    sim_values = []
    for i in range(len(EPReportName)):
        value = EnergyModel.Query(sim_model, EPReportName[i], EPReportForString[i], EPTableName[i], EPColumnName[i], [EPRowName[i]], EPUnits[i])
        sim_values.append(value)

    dict_names = [
        'total_source_energy_consumption_GJ',
        'total_site_energy_consumption_GJ',
        'total_source_energy_consumption_per_surface_MJ/m2',
        'total_site_energy_consumption_per_surface_MJ/m2',
        'site_energy_heating_GJ',
        'site_energy_cooling_GJ',
        'site_energy_heating_per_surface_MJ/m2',
        'site_energy_cooling_per_surface_MJ/m2',
        'total_exterior_wall_amount',
        'total_window_amount',
        'total_wall_area_m2',
        'wall_area_north_m2',
        'wall_area_east_m2',
        'wall_area_south_m2',
        'wall_area_west_m2',
        'total_window_opening_area_m2',
        'window_opening_area_north_m2',
        'window_opening_area_east_m2',
        'window_opening_area_south_m2',
        'window_opening_area_west_m2',
        'total_window-wall_ratio_%',
        'window-wall_ratio_north_%',
        'window-wall_ratio_east_%',
        'window-wall_ratio_south_%',
        'window-wall_ratio_west_%'
    ]
    ccomplex_dictionary = Topology.Dictionary(Apartment)
    ccomplex_sim_dictionary = Dictionary.ByKeysValues(dict_names, sim_values)
    ccomplex_merged_dicts = Dictionary.ByMergedDictionaries([ccomplex_dictionary, ccomplex_sim_dictionary])
    ccomplex_with_dicts = Topology.SetDictionary(Apartment, ccomplex_merged_dicts)
    tmp.cleanup()
    return ccomplex_with_dicts
    
def add_apertures_energy_sim(shell):

    with open('./graph/quantiles_dict.json', 'r') as qd:
        quantiles_dict = json.load(qd)
        
    with open('./graph/conversion_list.json', 'r') as cl:
        conversion_list = json.load(cl)

    total_area = 0
    room_names = []
    cells = []
    for index, face in enumerate(Topology.SubTopologies(shell, subTopologyType='face')):
        cell = Cell.ByThickenedFace(face, thickness=2.7, bothSides=False, reverse=False)
        room_name = Dictionary.ValueAtKey(Topology.Dictionary(face), 'type')
        room_names.append(room_name)
        surface = Face.Area(face, 4)
        total_area += surface
        ep_type_name = room_name[:-2].capitalize()
        
        function = ep_type_name.lower()
        quantile_list = quantiles_dict[function]
        if surface <= quantile_list[0]:
            size_string = 'xxs'
        elif surface <= quantile_list[1]:
            size_string = 'xs'
        elif surface <= quantile_list[2]:
            size_string = 's'
        elif surface <= quantile_list[3]:
            size_string = 'm'
        elif surface <= quantile_list[4]:
            size_string = 'l'
        elif surface <= quantile_list[5]:
            size_string = 'xl'
        elif surface > quantile_list[5]:
            size_string = 'xxl'
        label_string = function + '_' + size_string
        label = conversion_list.index(label_string)
        
        dicts = Dictionary.ByKeysValues(['id', 'area', 'element', 'ep_type', 'type', 'label'], [index, surface, 'room', ep_type_name, room_name, label])
        cell_with_dicts = Topology.SetDictionary(cell, dicts)
        cells.append(cell_with_dicts)

    topology_center = Topology.Centroid(shell)
    cellcomplex_variants = {'0' : CellComplex.ByCells(cells)}
    for rotation in range(0, 360, 90):
        if rotation:
            rotated_cells = [Topology.Rotate(cell, topology_center, x=0, y=0, z=1, degree=rotation) for cell in cells]
            cellcomplex_variants[str(rotation)] = CellComplex.ByCells(rotated_cells)

    for i, rotation in enumerate(cellcomplex_variants):
        ccomplex = cellcomplex_variants[rotation]
        extFaces = CellComplex.Decompose(ccomplex)['externalVerticalFaces']
        ccomplex_list = []
        for f in range(4):
            windows_one_direction = []
            windows_other_direction = []
            for t, face in enumerate(extFaces):
                np.random.seed(i + f + t)
                center = Topology.Centroid(face)
                x = y = z = np.random.uniform(0.2, 0.8)
                size = Face.Area(face)
                if size >= 2:
                    new_face = Topology.Scale(face, center, x, y, z)
                    value = Face.FacingToward(new_face, [0, 1, 0])
                    if value:
                        windows_one_direction.append(new_face)
                    else:
                        windows_other_direction.append(new_face)

            np.random.seed(i + f)           
            if np.random.randint(2):
                windows = windows_one_direction + windows_other_direction
            else:
                if np.random.randint(2):
                    windows = windows_one_direction
                    if not windows:
                        windows = windows_other_direction
                else:
                    windows = windows_other_direction
                    if not windows:
                        windows = windows_one_direction

            windows_with_dicts = []
            for index, window in enumerate(windows):
                surface = Face.Area(window)
                normal = Face.NormalAtParameters(window, outputType="xy")
                normal_degree_pi = np.arctan2(normal[0], normal[1])
                degree = np.mod(np.degrees(normal_degree_pi), 360)
                if degree >= 337.5 or degree < 22.5:
                    orientation_name = 'n'
                elif degree >= 22.5 and degree < 67.5:
                    orientation_name = 'ne'
                elif degree >= 67.5 and degree < 112.5:
                    orientation_name = 'e'
                elif degree >= 112.5 and degree < 157.5:
                    orientation_name = 'se'
                elif degree >= 157.5 and degree < 202.5:
                    orientation_name = 's'
                elif degree >= 202.5 and degree < 247.5:
                    orientation_name = 'sw'
                elif degree >= 247.5 and degree < 292.5:
                    orientation_name = 'w'
                elif degree >= 292.5 and degree < 337.5:
                    orientation_name = 'nw'
                    
                quantile_list = quantiles_dict['window']
                if surface <= quantile_list[0]:
                    size_string = 'xxs'
                elif surface <= quantile_list[1]:
                    size_string = 'xs'
                elif surface <= quantile_list[2]:
                    size_string = 's'
                elif surface <= quantile_list[3]:
                    size_string = 'm'
                elif surface <= quantile_list[4]:
                    size_string = 'l'
                elif surface <= quantile_list[5]:
                    size_string = 'xl'
                elif surface > quantile_list[5]:
                    size_string = 'xxl'
                label_string = 'window_' + size_string + '_' + orientation_name
                label = conversion_list.index(label_string)
                
                window_dict = Dictionary.ByKeysValues(
                    ['id', 'area', 'type', 'element', 'orientation', 'label'],
                    [index, surface, 'exterior', 'window', orientation_name, label])
                window = Topology.SetDictionary(window, window_dict)
                windows_with_dicts.append(window)

            ccomplex_with_apertures = Topology.AddApertures(Topology.Copy(ccomplex), windows_with_dicts, exclusive=True, subTopologyType='Face')
            id_str = str(i) + str(f)
            ccopmlex_dict = Dictionary.ByKeysValues(
                ['id', 'room_types', 'room_amount', 'surface', 'element'],
                [id_str, room_names, len(room_names), total_area, 'apartment'])
            ccomplex_with_apertures_dict = Topology.AddDictionary(ccomplex_with_apertures, ccopmlex_dict)
            ccomplex_with_sim_results = energy_simulation(ccomplex_with_apertures_dict, id_str)
            graph = Graph.ByTopology(ccomplex_with_sim_results, toExteriorApertures=True, useInternalVertex=False)
            ccdict = Dictionary.PythonDictionary(Topology.Dictionary(ccomplex_with_sim_results))
            consumption = ccdict['total_site_energy_consumption_per_surface_MJ/m2']
            energy_class = energy_to_class(consumption, quantiles_dict['energy'])
            ccomplex_list.append((ccomplex_with_sim_results, graph, energy_class, consumption))
        cellcomplex_variants[rotation] = ccomplex_list
    return cellcomplex_variants
    
def save_geometry_batch(ccomplex_database, name):
    tmp = tempfile.TemporaryDirectory()
    geometry_batch_path = os.path.join(tmp.name, name)
    os.mkdir(geometry_batch_path)
    for rotation, ccomplex_variant_list in enumerate(ccomplex_database.values()):
        rotation_path = os.path.join(geometry_batch_path, str(rotation))
        os.mkdir(rotation_path)
        for variant, ccomplex_tuple in enumerate(ccomplex_variant_list):
            Topology.ExportToJSONMK1(ccomplex_tuple[0], os.path.join(rotation_path, (str(variant) + '.json') ), overwrite=True)
    directory = pathlib.Path(geometry_batch_path)
    zip_path_name = './graph/data/geometry/geometry_batch_' + name + '.zip'
    with zipfile.ZipFile(zip_path_name,'w', zipfile.ZIP_DEFLATED) as zip:
            for file_path in directory.rglob("*"):
                zip.write(file_path, arcname=file_path.relative_to(directory))
    tmp.cleanup()
