import json
import numpy as np
import random
import imageio
import matplotlib.pyplot as plt
import networkx as nx
import math
import requests

random.seed(100)
np.random.seed(50)

def cal_distance(path):
    dis = 0
    for i in range(len(path) - 1):
        dis += D[path[i]][path[i + 1]]
    return dis

def plot_graph(adjacency_matrix, figure_title=None, print_shortest_path=False, src_node=None, filename=None,
               added_edges=None, pause=False):
    adjacency_matrix = np.array(adjacency_matrix)
    rows, cols = np.where(adjacency_matrix > 0)
    edges = list(zip(rows.tolist(), cols.tolist()))
    values = [adjacency_matrix[i][j] for i, j in edges]
    weighted_edges = [(e[0], e[1], values[idx]) for idx, e in enumerate(edges)]
    plt.cla()
    fig = plt.figure(1)
    if figure_title is None:
        plt.title("The shortest path for every node to the target")
    else:
        plt.title(figure_title)
    G = nx.Graph()
    G.add_weighted_edges_from(weighted_edges)
    # plot
    labels = nx.get_edge_attributes(G, 'weight')
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos=pos, with_labels=True, font_size=15)  # set with_labels to False if use node labels
    nodes = nx.draw_networkx_nodes(G, pos, node_color="y")
    nodes.set_edgecolor('black')
    nodes = nx.draw_networkx_nodes(G, pos, nodelist=[0, src_node] if src_node else [0], node_color="g")
    nodes.set_edgecolor('black')
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels, font_size=15)
    if print_shortest_path:
        print("The shortest path (dijkstra) is showed below: ")
        added_edges = []
        for node in range(1, num_nodes):
            shortest_path = nx.dijkstra_path(G, node, 0)  # [1,0]
            print("{}: {}".format("->".join([str(v) for v in shortest_path]),
                                  nx.dijkstra_path_length(G, node, 0)))
            added_edges += list(zip(shortest_path, shortest_path[1:]))
    if added_edges is not None:
        nx.draw_networkx_edges(G, pos, edgelist=added_edges, edge_color='r', width=2)

    if filename is not None:
        plt.savefig(filename)

    if pause:
        plt.pause(0.3)
    else:
        plt.show()

    # return img for video generation
    img = None
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = img.reshape((h, w, 3))
    return img


def get_best_actions(D, states):
    best_actions = []
    for node in range(1, num_nodes):
        actions = [(idx, states[idx]) for idx, weight in enumerate(D[node]) if weight > 0]
        actions, scores = zip(*actions)
        best_actions.append((node, actions[scores.index(max(scores))]))
    return best_actions

def print_best_actions(best_actions):
    # Convert indices to bus stop codes
    bus_stop_codes = [index_decoder[idx] for idx in best_actions]
    
    best_actions_info = []
    for i in range(len(bus_stop_codes) - 1):
        from_stop = bus_stop_codes[i]
        to_stop = bus_stop_codes[i + 1]
        
        # Determine which bus service to take
        bus_service = None
        for service_no, stops in included_services.items():
            stop_codes = [stop[1] for stop in stops]
            if from_stop in stop_codes and to_stop in stop_codes:
                bus_service = service_no
                break
        
        best_actions_info.append(f"{from_stop} -> {to_stop} via Bus {bus_service}")
    
    return ", ".join(best_actions_info)


def epsilon_greedy(s_curr, q, epsilon):
    if s_curr == terminal_state:
        return s_curr
    potential_next_states = np.where(np.array(D[s_curr]) > 0)[0]
    if len(potential_next_states) == 0:
        raise ValueError(f"No potential next states from current state {s_curr}.")
    if random.random() > epsilon:  # greedy
        q_of_next_states = q[s_curr][potential_next_states]
        s_next = potential_next_states[np.argmax(q_of_next_states)]
    else:  # random select
        s_next = random.choice(potential_next_states)
    return s_next


def q_learning(start_state=3, num_epoch=200, gamma=0.8, epsilon=0.05, alpha=0.1, visualize=True, save_video=False):
    print("-" * 20)
    print("q_learning begins ...")
    if start_state == terminal_state:
        raise Exception("start node(state) can't be target node(state)!")
    imgs = []  # useful for gif/video generation
    len_of_paths = []
    # init all q(s,a)
    q = np.zeros((num_nodes, num_nodes))  # num_states * num_actions
    for i in range(1, num_epoch + 1):
        s_cur = start_state
        path = [s_cur]
        len_of_path = 0
        while True:
            if s_cur == terminal_state:
                break
            try:
                s_next = epsilon_greedy(s_cur, q, epsilon=epsilon)
            except ValueError as e:
                print(e)
                return
            # greedy policy
            try:
                s_next_next = epsilon_greedy(s_next, q, epsilon=-0.2)  # epsilon<0, greedy policy
            except ValueError as e:
                print(e)
                return
            # update q
            reward = -D[s_cur][s_next]
            delta = reward + gamma * q[s_next, s_next_next] - q[s_cur, s_next]
            q[s_cur, s_next] = q[s_cur, s_next] + alpha * delta
            # update current state
            s_cur = s_next
            len_of_path += -reward
            path.append(s_cur)
        len_of_paths.append(len_of_path)
        if visualize:
            img = plot_graph(D, print_shortest_path=False, src_node=start_state,
                             added_edges=list(zip(path[:-1], path[1:])), pause=True,
                             figure_title="q-learning\n {}th epoch: {}".format(i, cal_distance(path)))
            imgs.append(img)
    if visualize:
        plt.show()
    if visualize and save_video:
        print("begin to generate gif/mp4 file...")
        imageio.mimsave("q-learning.gif", imgs, fps=5)  # generate video/gif out of list of images
    # print the best path for start state to target state
    strs = "best path for node {} to node 1: ".format(start_state)
    print(strs)
    print(path)
    best_action_path=print_best_actions(path)
    print(best_action_path)
    return 0

# Function to calculate distance in km between nodes using their lat long values without haversine package
def calculate_distance(node1, node2):
    R = 6371.0  # Radius of the Earth in km

    lat1, lon1 = node1
    lat2, lon2 = node2

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return round(distance, 2)

def get_lat_long(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['found'] > 0:
            result = data['results'][0]
            lat_long = [float(result['LATITUDE']), float(result['LONGITUDE'])]
            return lat_long
        else:
            print("No results found")
            return None
    else:
        print(f"Error fetching data: {response.status_code} - {response.reason}")
        return None


if __name__ == '__main__':

    # Load JSON data
    with open('bus_routes_dict.json', 'r') as f:
        bus_routes = json.load(f)
    
    current_location_postal= input("Please enter your current location postal code address: ")
    destination_postal= input("Please enter your destination postal code address: ")

    # Construct URLs with the user-provided postal codes
    url_current = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={current_location_postal}&returnGeom=Y&getAddrDetails=Y&pageNum=1"
    url_destination = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={destination_postal}&returnGeom=Y&getAddrDetails=Y&pageNum=1"

        # Get latitude and longitude for the current location
    current_lat_long = get_lat_long(url_current)
    if current_lat_long:
        print(f"Current Location Latitude and Longitude: {current_lat_long}")

    # Get latitude and longitude for the destination
    destination_lat_long = get_lat_long(url_destination)
    if destination_lat_long:
        print(f"Destination Latitude and Longitude: {destination_lat_long}")
    
    destination_lat_long = [round(coord, 2) for coord in destination_lat_long]
        
    # Replace with the bus service numbers you want to extract
    bus_service_numbers = ['243W', '243G', '179']  
    included_services = {service_no: bus_routes[service_no] for service_no in bus_service_numbers if service_no in bus_routes}
    
    num_epoch=100
    start_state=0
    #Use this to add more weights for the walking distance edge
    walking_factor=4
    
    # Collect all bus stop codes and find the destination node code
    destination_node_code = None
    all_stops = ['starting_node']  # Add starting node as the first entry
    
    for service in included_services.values():
        for stop in service:
            if stop[1] not in all_stops:
                all_stops.append(stop[1])
            stop_latlong = [round(coord, 2) for coord in stop[2:4]]
            if stop_latlong == destination_lat_long:
                destination_node_code = stop[1]
    
    print("Terminal_state: ", destination_node_code)

    # Create encoding and decoding dictionaries
    stop_indices = {stop: idx for idx, stop in enumerate(all_stops)}
    index_decoder = {idx: stop for stop, idx in stop_indices.items()}
    
    # Print the encoding dictionary
    print("\nEncoding Dictionary (stop_indices):")
    print(stop_indices)

    # Print the decoding dictionary
    print("\nDecoding Dictionary (index_decoder):")
    print(index_decoder)
    
    terminal_state= stop_indices[destination_node_code]
    
    # Initialize adjacency matrix
    D = np.zeros((len(all_stops), len(all_stops)))

    # Fill adjacency matrix with distances from bus routes
    for service in included_services.values():
        for i in range(len(service) - 1):
            from_stop = service[i][1]
            to_stop = service[i + 1][1]
            distance = service[i + 1][0]
            D[stop_indices[from_stop], stop_indices[to_stop]] = distance
    
    # Add additional edges between the starting node and all other bus stop nodes
    #Starting node is always index 0
    for stop, idx in stop_indices.items():
        if stop != 'starting_node':
            stop_latlong = [included_services[service][i][2:4] for service in included_services for i in range(len(included_services[service])) if included_services[service][i][1] == stop][0]
            distance = calculate_distance(current_lat_long, stop_latlong)
            if distance < 1:  # Only add edges with distance less than 1 km
                D[0, idx] = distance * walking_factor

    
    # Print the adjacency matrix
    print("Adjacency Matrix:\n", D)
    num_nodes = len(D)
    print("number of nodes: ", num_nodes)
    
    # Create graph from adjacency matrix
    G = nx.from_numpy_array(D)

    # Define node colors
    node_colors = ['red' if idx == 0 else 'blue' if idx == 1 else 'green' for idx in range(len(all_stops))]

    # Draw the graph
    plt.figure(figsize=(10, 8))  # Set figure size
    pos = nx.spring_layout(G)  # Positions for all nodes

    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=250, font_size=5, font_color='white')

    # Draw edge labels (weights)
    edge_labels = {(i, j): D[i, j] for i in range(len(D)) for j in range(len(D)) if D[i, j] > 0}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black',font_size=5)

    # Show plot
    plt.show()
    q_learning(start_state=start_state, num_epoch=num_epoch, gamma=0.8, epsilon=0.05, alpha=0.1, visualize=True, save_video=True)
