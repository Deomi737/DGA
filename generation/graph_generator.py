import os
import pydot
import numpy as np

from pydot import Node, Edge, Dot

# GNP Algorithm to creating a adjacency Matrix
def gnp(n, p):
    m = np.zeros((n, n))

    for i in range(n):
        for j in range(i):
            if np.random.random() < p:
                m[i][j] = 1
            else:
                m[i][j] = 0

    return m

# Return all graphs from a dot-file
def read(file_path: str) -> Dot:
    graphs = pydot.graph_from_dot_file(file_path)
    return graphs

# Return first graph from a dot-file
def read_first(file_path: str) -> Dot:
    #print("Reading Graph from file... ", end="", flush=True)
    graphs = pydot.graph_from_dot_file(file_path)
    graph = graphs[0]
    #print("Done")
    return graph

# Write a dot to a dot-file
def write(file_path: str, dot: Dot) -> None:
    directory_path = os.path.join(file_path.rsplit('/', 1)[0])
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    dot.write(file_path)

# ensure that all nodes have atlease one connection to a successor or predecessor
def ensure_connections(m):
    for i in range(len(m)):
        
        pred = 0
        j = 0
        while pred == 0 and j < i:
            pred += m[i][j]
            j += 1
        
        if pred == 0 and not i == 0:
            random_pred = int(np.random.randint(0, i))
            m[i][random_pred] = 1

        succ = 0
        j = i + 1
        while succ == 0 and j < len(m):
            succ += m[j][i]
            j += 1

        if succ == 0 and not i == len(m) - 1:
            random_succ = int(np.random.randint(i + 1, len(m)))
            m[random_succ][i] = 1

    return m

# Clean all edges a -> c, if a -> * -> c exists
def clean_transition_edges(m):
    succ = []
    # succeding nodes
    for i in range(len(m)):
        succ.append([])
        # get all direct successors and create lists
        for j in range(i + 1, len(m)):
            if m[j][i] == 1:
                succ[i].append(j)

    # check for transitive nodes from last to first
    for n in range(len(m) - 1, -1, -1):
        # calculate all succeding nodes of direct successors
        all_succeding = []
        for x in succ[n]:
            all_succeding.extend(succ[x])

        # Check if a direct successor is a successor of directly succeding nodes and delete the edge 
        for x in succ[n]:
            if x in all_succeding:
                m[x][n] == 0

        # add all none duplicates to succeding nodes 
        succ[n].extend(list(set(all_succeding) - set(succ[n]))) 
    
    return m

# Convert the precedence Constraints of a matrix to a pydot dot
def matrix_to_graph(matrix) -> Dot: #: List[List[int]]
    # initialize the Graph
    graph = Dot(graph_type='digraph')
    #graph.set_node_defaults(color='lightgray', style='filled', shape='box', fontname='Courier', fontsize='10')

    # create all nodes
    length = len(matrix)
    for i in range(length):
        node = Node(i)
        graph.add_node(node)
    
    # create all Edges
    for row in range(length):
        for column in range(length):
            if matrix[row][column] == 1:
                # ziemlich sicher dass es so rum muss
                edge = Edge(src=column, dst=row)
                graph.add_edge(edge)

    return graph

# Create graphs with gnp
def create_graphs_gnp(path, amount_tasks, amount_set, n_min, n_max, p_min, p_max, num_nodes=None):
    tasksets = []
    for s in range(amount_set):
        taskset = []
        for t in range(amount_tasks):
            if num_nodes == None:
                matrix = gnp(int(np.random.uniform(n_min, n_max)), np.random.uniform(p_min, p_max))
            else:
                matrix = gnp(int(sum(list(num_nodes[s][t]))), np.random.uniform(p_min, p_max))

            matrix = ensure_connections(matrix)
            matrix = clean_transition_edges(matrix)

            #write(f'{path}/{s:03d}/{t:03d}.dot', matrix_to_graph(matrix))

            taskset.append(matrix)

        #np.save(f'{path}/{s:03d}/taskset.npy', taskset)
        tasksets.append(taskset)
    
    directory_path = os.path.join(f"{path}/tasksets.npy".rsplit('/', 1)[0])
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    np.save(f'{path}/tasksets.npy', taskset)

    return tasksets