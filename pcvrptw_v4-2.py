# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 15:36:02 2023

@author: Hei Lap Man
@software{ortools,
  title = {OR-Tools},
  version = { v9.6 },
  author = {Laurent Perron and Vincent Furnon},
  organization = {Google},
  url = {https://developers.google.com/optimization/},
  date = { 2023-03-13 }
}
"""
#%%
import pandas as pd
import numpy as np

# Import data files
dff = pd.read_excel('traveltime_matrix.xlsx', sheet_name='walking', index_col=0)
demand_df = pd.read_excel('Homecare_Transport_2023b.xlsx', sheet_name='Current_Timed', index_col=0)
index_df = pd.read_csv('locations_index.csv', index_col=1)

# Filter demand_df for rows where Svc_Provider_ID is equal to 224
filtered_demand_df = demand_df[(demand_df['Svc_Provider_ID'] == 240) & (
    demand_df['Day'] == 'Sun') & (demand_df['S'] == 'A')]
df_temp = filtered_demand_df.copy()

# Reset index to use LSOA in merge operation
filtered_demand_df = filtered_demand_df.reset_index()

# Remove spaces from the 'Column' column
filtered_demand_df['LSOA'] = filtered_demand_df['LSOA'].str.replace(' ', '').str.strip()

# Merge with index_df to get the corresponding id for each LSOA
merged_df = filtered_demand_df.merge(index_df, how='left', left_on='LSOA', right_on='Name')
# Create a copy of merged_df and reset the index
merged_df_copy = merged_df.copy().reset_index()

# Create an empty DataFrame with the same index and columns as merged_df
result_matrix = pd.DataFrame(index=merged_df_copy.index, columns=merged_df_copy.index)

# Iterate over the index and columns of the new DataFrame
for i in result_matrix.index:
    for j in result_matrix.columns:
        # For each cell, use the 'id' value at the corresponding row in merged_df_copy
        from_id = merged_df_copy.loc[i, 'id']
        to_id = merged_df_copy.loc[j, 'id']
        result_matrix.loc[i, j] = dff.loc[from_id, to_id]

# output_file_path = "240_Sun_walk_A.csv"
# result_matrix.to_csv(output_file_path, index=False)

# Demand
demand = df_temp['Visit_Mins'].values.tolist()

# TW
# Define a function to convert time to minutes within a 30-minute window
def convert_to_minutes(time):
    hour, minute = map(int, time.split(':'))
    return hour * 60 + minute

# Convert the 'START_TIME' column to a list of minute ranges within a 30-minute window
time_list = df_temp['START_TIME'].apply(convert_to_minutes).apply(lambda x: (x-15, x+15)).tolist()

#%%
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# Read travel time matrix
df = pd.read_csv("240_Sun_walk_A.csv")
df.fillna(1e6, inplace=True) # Replace non-reachable places by large number

# Configure mode of transport
mode = 'walk'
if mode == 'cycle':
    df = df + 2 # buffer time for parking
    df = df.replace(0, 3) # buffer time for within LSOA travel
elif mode == 'walk':
    df = df.replace(0, 9)
elif mode == 'car':
    df = df + 2
    df = df.replace(0, 2)

# Add a row at index 0
df = pd.concat([pd.DataFrame([np.zeros(df.shape[1])], columns=df.columns), df]).reset_index(drop=True)

# Add a column at index 0
df.insert(0, 0, 0)
df.columns = range(df.shape[1])

time_list.insert(0, (0, 60*24)) # 7-14h(A) or 15-22h(P)
demand = [0] + demand # dummy depot costs no time
#%%
# New row to append
new_row = pd.DataFrame({
    'LSOA': [0],
    'Svc_Provider_ID': [0],
    'Carer': [0],
    'Day': [0],
    'START_TIME': [0],
    'S': [0],
    'Visit_Mins': [0],
    'Visit_Modif': [0]
}, index=[0])

# Append new row to dataframe
filtered_demand_df = pd.concat([new_row, filtered_demand_df]).reset_index(drop=True)

# Group the DataFrame by the columns that define a visit
grouped_demand_df = filtered_demand_df.groupby(["LSOA", "Svc_Provider_ID", "Day", "START_TIME", "S", "Visit_Mins", "Visit_Modif", "LT_ST"])

# Initialize an empty list to store the pairs
visit_pairs = []

# Iterate over the groups
for name, group in grouped_demand_df:
    # If this group has more than one row, it is a double-up visit
    if len(group) > 1:
        # Check if all rows in this group have the same role
        if group["Carer"].nunique() > 1:
            # Get the indices of the rows in this group and add them to the pairs list
            visit_pairs.append(tuple(group.index.tolist()))
visit_pairs = [t for t in visit_pairs if len(t) > 1]

new_visit_tuples = []
for t in visit_pairs:
    if len(t) == 4:
        new_visit_tuples.append((t[0], t[1]))
        new_visit_tuples.append((t[2], t[3]))
    else:
        new_visit_tuples.append(t)

visit_pairs = new_visit_tuples

#%%
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['time_matrix'] = [[int(num) for num in sublist] for sublist in df.values.tolist()]
    data['time_windows'] = time_list
    data['scores'] = demand
    data['num_vehicles'] = 30 # carers
    data['depot'] = 0
    data['max_shift_time'] = 60*7
    data['min_shift_time'] = 0
    data['endshift'] = 60*15
    data['max_travel_time'] = int(data['max_shift_time']*1)
    data['visit_pairs'] = visit_pairs
    # First Solution Strategies
    data['first_solution_strategies'] = [
        routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC,
        routing_enums_pb2.FirstSolutionStrategy.BEST_INSERTION,
        routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC,
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
    ]
    # Local Search Metaheuristics
    data['local_search_strategies'] = [
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
        routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
    ]
    return data

results_dict = {}
def main():
    """Entry point of the program."""
    global results_dict
    
    # Instantiate the data problem.
    data = create_data_model()
    num_nodes = len(data['time_matrix'])
    with open ('result.txt', 'a') as f:
        print(f'Num nodes = {num_nodes}', file=f)
    
    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(num_nodes,
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback
    def traveltime_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(traveltime_callback)

    # Define cost of each arc
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Create and register a visit time callback
    def visittime_callback(from_index, to_index):
        """Returns the travel plus visit time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node] + data['scores'][from_node]

    visit_callback_index = routing.RegisterTransitCallback(visittime_callback)
    
    # Limit shift time.
    routing.AddDimension(
        visit_callback_index,
        0,  # no slack
        data['max_shift_time'],  # carer maximum shift time
        True,  # start cumul to zero
        'VisitTime')
    shift_dimension = routing.GetDimensionOrDie('VisitTime')

    for vehicle_id in range(data['num_vehicles']):
        end_index = routing.End(vehicle_id)
        shift_dimension.SetCumulVarSoftLowerBound(end_index, data['min_shift_time'], 0)

    # Limit travel time.
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        data['max_travel_time'],  # carer maximum travel time
        True,  # start cumul to zero
        'TravelTime')
    traveltime_dimension = routing.GetDimensionOrDie('TravelTime')
    traveltime_dimension.SetGlobalSpanCostCoefficient(1)
    
    # Allow to drop nodes.
    for node in range(1, num_nodes):
        routing.AddDisjunction(
                [manager.NodeToIndex(node)],
                data['scores'][node])

    # Add Time Windows constraint
    time = 'Time'
    routing.AddDimension(
        visit_callback_index,
        999,  # allow waiting time
        data['endshift'],  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time)
    time_dimension = routing.GetDimensionOrDie(time)
    # Add time window constraints for each location except depot.
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == data['depot']:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
    # Add time window constraints for each vehicle start node.
    depot_idx = data['depot']
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(
            data['time_windows'][depot_idx][0],
            data['time_windows'][depot_idx][1])
    for i in range(data['num_vehicles']):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i)))
        
    # Add double up visit constraints
    for visit in data['visit_pairs']:
        if len(visit) == 2:
            index1 = manager.NodeToIndex(visit[0])
            index2 = manager.NodeToIndex(visit[1])
            # Add a synchronization constraint between the two visit node
            routing.solver().Add(time_dimension.CumulVar(index1) == time_dimension.CumulVar(index2))
        elif len(visit) == 3:
            index1 = manager.NodeToIndex(visit[0])
            index2 = manager.NodeToIndex(visit[1])
            index3 = manager.NodeToIndex(visit[2])
            routing.solver().Add(time_dimension.CumulVar(index1) == time_dimension.CumulVar(index2))
            routing.solver().Add(time_dimension.CumulVar(index1) == time_dimension.CumulVar(index3))
    
    # Setting first solution heuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        data['first_solution_strategies'][2])
    # Setting local search strategy
    search_parameters.local_search_metaheuristic = (
        data['local_search_strategies'][2])
    search_parameters.solution_limit = 500
    search_parameters.time_limit.seconds = 12000
    search_parameters.log_search = True

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        results_dict = print_solution(data, manager, routing, solution, 'vrp_result.csv')

#%%
# Create a console solution printer.
import csv
from datetime import datetime

def print_solution(data, manager, routing, solution, filename):
    """Write the solution to a CSV file and return as a dictionary"""
    
    results = {}  # This will hold our results in dictionary format
    
    # Summary
    results['Summary'] = {}
    results['Summary']['Execution_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results['Summary']['Objective'] = solution.ObjectiveValue()
    
    # Dropped nodes
    total_dropped = 0
    results['Dropped_nodes'] = []
    for node in range(routing.Size()):
        if routing.IsStart(node) or routing.IsEnd(node):
            continue
        if solution.Value(routing.NextVar(node)) == node:
            results['Dropped_nodes'].append(manager.IndexToNode(node))
            total_dropped += 1
    results['Summary']['Dropped_num'] = total_dropped
    
    # Routes
    results['Routes'] = []
    time_dimension = routing.GetDimensionOrDie('Time')
    
    total_score_collected = 0
    total_travel_time = 0
    total_shift_time = 0

    for v in range(data['num_vehicles']):
        route_info = {}
        
        index = routing.Start(v)
        route_travel_time = 0
        route_score = 0
        route_nodes = []
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_nodes.append(node_index)
            route_score += data['scores'][node_index]
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_travel_time += routing.GetArcCostForVehicle(previous_index, index, 0)
            
        route_info['Carer'] = v
        route_info['Route'] = route_nodes
        route_info['Score'] = route_score
        route_info['Travel Time'] = route_travel_time
        route_info['Shift Time'] = route_score + route_travel_time
        
        results['Routes'].append(route_info)

        total_score_collected += route_score
        total_travel_time += route_travel_time
        total_shift_time += (route_score + route_travel_time)

    results['Summary']['T_score'] = total_score_collected
    results['Summary']['T_traveltime'] = total_travel_time
    results['Summary']['T_shifttime'] = total_shift_time
    results['Summary']['TU'] = total_travel_time/ total_shift_time
    
    # Write to CSV
    with open(filename, 'a', newline='') as csvfile:
        writer_summary = csv.DictWriter(csvfile, fieldnames=results['Summary'].keys())
        writer_summary.writeheader()
        writer_summary.writerow(results['Summary'])
        
        csvfile.write('\n')
        
        writer_routes = csv.DictWriter(csvfile, fieldnames=results['Routes'][0].keys())
        writer_routes.writeheader()
        for route in results['Routes']:
            writer_routes.writerow(route)
    
    return results

#%%    
import time, winsound
start_time = time.time()

if __name__ == '__main__':
    main()
    
end_time = time.time()
execution_time = end_time - start_time  # Calculate the difference

with open ('result.txt', 'a') as f:
    print(f"The execution time of the program is: {execution_time} seconds", file=f)
    
winsound.Beep(1000, 2000)
print("Routing Model has finished running!")

#%%
# Human-understandable color names
colors = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "lightblue", "lightgreen", 
    "darkblue", "darkgreen", "cyan", "magenta", "lavender", "beige", "gray", "black", "olive", "teal",
    "aqua", "maroon", "gold", "silver", "plum", "ivory", "indigo", "coral", "lime", "tan"]

# Extracting the coordinates from the "coordinates" column, accounting for parentheses
route_coordinates_final = []
merged_df['coordinates'] = merged_df['geometry'].str.extract(r'POINT \(([-\d.]+) ([\d.]+)\)').astype(float).apply(tuple, axis=1)

for route in range(len(results_dict['Routes'])):
    coords = []
    # Skip index 0
    for index in results_dict['Routes'][route]['Route'][1:]:
        lon, lat = merged_df.loc[index - 1, 'coordinates']
        lon, lat = float(lon), float(lat)
        coords.append([lat, lon])
    route_coordinates_final.append({"coords": coords, "color": colors[route]})

route_coordinates_final[:3] # Display the first 3 routes

#%%
# Convert the format of the routes to match the desired format

formatted_routes = []

for route in range(len(route_coordinates_final)):
    transformed_coords = [[coord[1], coord[0]] for coord in route_coordinates_final[route]['coords']]
    formatted_route = {
        'coordinates': transformed_coords
    }
    formatted_routes.append(formatted_route)

#%%
import requests
import folium
import json
import polyline
from folium.plugins import HeatMap, PolyLineOffset

headers = {
    'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
    'Authorization': '5b3ce3597851110001cf6248e4ae8827627248329c6577e65016c07b',
    'Content-Type': 'application/json; charset=utf-8'
}


# Initialise the map
m = folium.Map(location=[52.414710, -1.774300], zoom_start=13)
n = folium.Map(location=[52.414710, -1.774300], zoom_start=13)

for r in range(len(formatted_routes)):
    print(f"Starting route {r}...")
    call = requests.post('https://api.openrouteservice.org/v2/directions/foot-walking', json=formatted_routes[r], headers=headers)
    print(call.status_code, call.reason)
    
    # Parse the JSON string into a Python dictionary
    data = json.loads(call.text)
    
    # Extracting the encoded polyline string
    encoded_str = data["routes"][0]["geometry"]
    
    # Decode the encoded polyline string to a list of coordinates
    decoded_coords = polyline.decode(encoded_str)
    folium.PolyLineOffset(decoded_coords, color=route_coordinates_final[r]['color'], weight=3.5, opacity=1, offset=-5)\
        .add_child(folium.Popup(r))\
        .add_to(m)
        
    for p in range(len(route_coordinates_final[r]['coords'])):
        tooltip_content = "<i>{}-{}</i>".format(r+1, p+1)
        folium.Marker(route_coordinates_final[r]['coords'][p], tooltip=tooltip_content).add_to(m)
        print(f"Added point {p} to the map.")
    print(f"Added route {r} to the map.")

heat_data = []
for r in range(len(route_coordinates_final)):
    for p in range(len(route_coordinates_final[r]['coords'])):    
        heat_data += [route_coordinates_final[r]['coords'][p]]

HeatMap(heat_data).add_to(n)
        
# Display the map
m.save("map.html")
n.save("heatmap.html")

winsound.Beep(1000, 2000)
print("Maps have been created!")

#%%
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
