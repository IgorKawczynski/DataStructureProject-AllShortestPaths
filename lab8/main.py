from enum import Enum
from typing import Optional, Callable
from typing import Dict
from typing import List
from typing import Any
import math
import matplotlib.pyplot as plt

import networkx as nx
from graphviz import Digraph


# ---------------------------------------------- NODE ----------------------------------------------- #


class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None


# -------------------------------------------- LINKEDLIST -------------------------------------------- #
class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def push(self, data: Any) -> None:
        new_node = Node(data)
        if self.head is None:
            self.tail = new_node
        new_node.next = self.head
        self.head = new_node

    def append(self, element: Any) -> None:
        new_node = Node(element)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        self.tail.next = new_node
        self.tail = new_node

    def node(self, at: int) -> Node:
        temp = self.head
        if temp is None:
            return None
        for i in range(at):  # powinno byc raczej at-1, wtedy zwroci wezeł o odpowiednim indeksie
            temp = temp.next
        return temp

    def insert(self, data: Any, after: Node) -> None:
        if after is None:
            print("There is no such node in this linkedList")
            return None
        new_node = Node(data)
        if after == self.tail:
            after.next = new_node
            self.tail = new_node
        new_node.next = after.next
        after.next = new_node

    def pop(self) -> Any:
        if self.head == 0:
            return 0
        removed = self.head
        removed.data = self.head.data
        self.head = self.head.next
        return removed.data

    def remove_last(self) -> Any:
        if self.head == 0:
            return 0
        temp = self.head
        while temp.next.next is not None:
            temp = temp.next
        self.tail = temp
        temp.next = None  # ustawienie nastepnika na 0, koniec listy
        return temp.data

    def remove(self, after: Node) -> None:
        if after is None:
            print("There is no such node in this linkedList")
            return None
        else:
            self.tail = after
            after.next = None

    def __str__(self) -> str:
        temp = self.head
        temp_list = ""
        if temp is None:
            print("List is empty")
        while temp is not None:
            if temp.next is not None:
                temp_list = temp_list + str(temp.data) + ' -> '  # do ogona dodaje strzalke
            else:
                temp_list = temp_list + str(temp.data)  # dla ogona nie dodaje strzalki
            temp = temp.next
        return temp_list

    def __len__(self) -> int:
        current = self.head
        sum_len = 0
        if current is None:
            return 0
        while current is not None:
            sum_len = sum_len + 1
            current = current.next
        return sum_len


# ---------------------------------------------- QUEUE ---------------------------------------------- #
class Queue:
    _storage: LinkedList

    def __init__(self) -> None:
        self._storage = LinkedList()

    def peek(self) -> Any:
        if self._storage == 0:
            return 0
        return self._storage.head.data

    def enqueue(self, element: Any) -> None:
        return self._storage.append(element)

    def dequeue(self) -> Any:
        return self._storage.pop()

    def __len__(self) -> int:
        return len(self._storage)

    def __str__(self) -> str:
        temp_queue = ""
        if self._storage is None:
            print("Queue is empty")
        for i in range(len(self._storage)):
            if i == len(self._storage) - 1:
                temp_queue = temp_queue + str(self._storage.node(i).data)  # dla ostatniego elementu - brak przecinka
            else:
                temp_queue = temp_queue + str(self._storage.node(i).data) + ", "
        return temp_queue


# ---------------------------------------- GRAPH ATTRIBUTES ------------------------------------------ #
class EdgeType(Enum):
    directed = 1
    undirected = 2


class Vertex:
    data: Any
    index: int

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __repr__(self):
        return self.data


class Edge:
    source: Vertex
    destination: Vertex
    weight: Optional[float]

    def __init__(self, source, destination, weight):
        self.source = source
        self.destination = destination
        self.weight = weight

    def __repr__(self):
        return "{}: v{}".format(self.destination.data, self.destination.index)


# ------------------------------------------- GRAPH  -------------------------------------------- #


class Graph:
    adjacencies: Dict[Vertex, List[Edge]]

    def __init__(self):
        self.adjacencies = dict()

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(list(self.adjacencies.keys())):
            result = list(self.adjacencies.keys())[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    def create_vertex(self, data):
        self.adjacencies[Vertex(data, len(self.adjacencies))] = list()
        # # adding key
        # self.adjacencies.pop(Vertex(data, len(self.adjacencies)))
        # # adding value to the key
        # self.adjacencies[Vertex(data, len(self.adjacencies))] = list()
        # return Vertex(data, len(self.adjacencies))

    def get_vertex(self, data):
        vertex = None
        for vertex in self.adjacencies.keys():
            if vertex.data == data:
                return vertex
        if vertex is None:
            return False

    def get_all_list_of_vertexes(self) -> List[Vertex]:
        graph_vertices_keys1 = [x for x in self.adjacencies.keys()]
        return graph_vertices_keys1

    def get_all_neighbours(self) -> List[List[Edge]]:
        graph_vertices_vals1 = [x for x in self.adjacencies.values()]
        return graph_vertices_vals1

    def get_neighbours(self, key_temp) -> List[Edge]:
        return self.adjacencies.get(key_temp)

    def get_adjacency(self):
        return self.adjacencies

    def add_directed_edge(self, source: Vertex, destination: Vertex, weight: Optional[float] = None):
        self.adjacencies[source].append(Edge(source, destination, weight))

    def add_undirected_edge(self, source: Vertex, destination: Vertex, weight: Optional[float] = None):
        # using method add_directed_edge in reversion ( <- -> )
        self.add_directed_edge(source, destination, weight)
        self.add_directed_edge(destination, source, weight)

    def add(self, edge: EdgeType, source: Vertex, destination: Vertex, weight: Optional[float] = None):
        if source not in self.adjacencies.keys():
            raise ValueError("There is no such Source Vertex in this Graph, "
                  "try another one!, there is list of available vertexes : " + self.adjacencies.keys().__str__())
        elif destination not in self.adjacencies.keys():
            raise ValueError("There is no such Destination Vertex in this Graph, "
                  "try another one!, there is list of available vertexes :" + self.adjacencies.keys().__str__())
        else:
            if edge == EdgeType.directed:
                self.add_directed_edge(source, destination, weight)
            elif edge == EdgeType.undirected:
                self.add_undirected_edge(source, destination, weight)

    def traverse_bfs(self, visit: Callable[[Any], None]):
        queue1 = Queue()
        visited = []
        start = list(self.adjacencies.keys())[0]  # rzutowanie do listy, aby uzyskac 1wszy element(klucz) ze słownika
        queue1.enqueue(start)

        while len(queue1) is not None:
            v = queue1.dequeue()
            visit(v)
            for edge in self.adjacencies[v]:  # for each edge that is neighbour to V
                if edge.destination not in visited:
                    queue1.enqueue(edge.destination)
                    visited.append(edge.destination)

    def traverse_depth_first(self, visit: Callable[[Any], None]):
        first = list(self.adjacencies.keys())[0]
        visited = list()
        v = first
        self.dfs(v, visited, visit)

    def dfs(self, v: Vertex, visited: List[Vertex], visit: Callable[[Any], None]):
        visit(v)
        visited.append(v)
        for edge in self.adjacencies[v]:  # for each edge that is neighbour to v
            if edge.destination not in visited:
                self.dfs(edge.destination, visited, visit)  # in that way we move to every single vertex in this Graph

# Overriding repr method ( for adequate print )
#     def __repr__(self):
#         temp = ""
#         listVertex = list(self.adjacencies.keys())
#         for vertex in listVertex:
#             temp += f'{vertex.data} ----> '
#             edges = self.adjacencies[vertex]
#             neighbours = list()
#             for edge in edges:
#                 neighbours.append(edge.destination)
#             temp += f'{neighbours}\n'
#         return temp

    def __str__(self):
        temp = ""
        for vertexes in self.adjacencies:
            temp = temp + "{}: v{} ----> {} \n".format(vertexes.data, vertexes.data, self.adjacencies[vertexes])
        return temp

    def show(self, path=[]):
        edges = []
        # setting that graph is not weighted at first
        weighted = False

        for vertex in self:
            edges.extend(self.adjacencies[vertex])
        G = nx.DiGraph()
        nodes = []

        for p1 in path:
            for edgez in p1:
                if nodes.count(edgez) is None:
                    nodes.append(edgez)

        for edge in edges:
            if edge.weight is not None:
                weighted = True
                # creating edges and attributes for them, color, weight label, size(width) of em'
                if(edge.source, edge.destination) in path:
                    G.add_edge(edge.source, edge.destination, weight=edge.weight, color='gold', width=5) # from doc : nodes are automatically added
                else:
                    G.add_edge(edge.source, edge.destination, weight=edge.weight, color='black', width=3)
            else:
                if(edge.source, edge.destination) in path:
                    G.add_edge(edge.source, edge.destination, color='gold', width=5)
                else:
                    G.add_edge(edge.source, edge.destination, color='black', width=3)

        # from doc, getting colours and widths
        colors = nx.get_edge_attributes(G, 'color').values()
        widths = [x for x in nx.get_edge_attributes(G, 'width').values()]
        # position schema for graphs, if weighted then planar, if not, circular
        if weighted:
            pos = nx.planar_layout(G)
        else:
            pos = nx.circular_layout(G)

        nx.draw(G, pos, node_color='black', edgecolors='silver', edge_color=colors, width=widths,
                with_labels=True, node_size=1000, alpha=0.85, arrows=True, arrowsize=8)

        # doc : networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx_edges.html

        # edges and nodes were added, but now we need labels on them and more attributes
        labels = nx.get_edge_attributes(G, 'weight')

        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        nx.draw_networkx_labels(G, pos, font_size=12, font_color="silver")

        plt.show()


traversed_vertex_list = []
# using method ,, visit '' to append elements in ,, traversed_vertex_list '' to print how traversion worked


def visit(vertex):
    traversed_vertex_list.append(vertex.data)


# CREATING THE GRAPH
graph1 = Graph()
graph1.create_vertex("0")
graph1.create_vertex("1")
graph1.create_vertex("2")
graph1.create_vertex("3")
graph1.create_vertex("4")
graph1.create_vertex("5")

graph1.add(EdgeType(2), graph1.get_vertex("0"), graph1.get_vertex("1"), 5)
graph1.add(EdgeType(2), graph1.get_vertex("0"), graph1.get_vertex("3"), 4)
graph1.add(EdgeType(2), graph1.get_vertex("0"), graph1.get_vertex("5"), 84)
graph1.add(EdgeType(2), graph1.get_vertex("1"), graph1.get_vertex("3"), 11)
graph1.add(EdgeType(2), graph1.get_vertex("1"), graph1.get_vertex("2"), 6)
graph1.add(EdgeType(2), graph1.get_vertex("2"), graph1.get_vertex("3"), 10)
graph1.add(EdgeType(2), graph1.get_vertex("4"), graph1.get_vertex("5"), 7)
graph1.add(EdgeType(2), graph1.get_vertex("5"), graph1.get_vertex("2"), 4)


# print("\n-----------------------KLASA GRAPH----------------------- : ")
# print("\nDISPLAYING ALL VERTICES ( AS THEIR DATA ) : ")  # as data, because representation of it is data
# print(graph1.get_all_list_of_vertexes())
#
# print("\nDISPLAYING VERTICES WITH THEIR INDEXES AND THEIR ADJACENCY LIST")
# print(graph1)
#
# print("\nGETTING THE NEIGHBOURS OF GIVEN VERTEX IN GRAPH : ")
# print(graph1.get_neighbours(graph1.get_vertex("0")))
#
# print("\nGETTING THE ADJACENCY LIST IN GRAPH : ")
# print(graph1.get_adjacency())
#
# # USING TRAVERSE DEPTH FIRST
# graph1.traverse_depth_first(visit)
#
# print("\nPRINTING THE LIST OF ALL TRAVERSED VERTEXES :")
# print(traversed_vertex_list)
#
# # CLEARING THE LIST
# traversed_vertex_list.clear()
#
# # DISPLAYING THE GRAPH
# graph1.show()


# ------------------------------------------ GRAPHPATH  ------------------------------------------- #

class GraphPath:
    graph: Graph
    source: Vertex
    destination: Vertex
    path: list

    def __init__(self, graph: Graph, source: Vertex, destination: Vertex):

        self.graph = graph

        if source and destination in self.graph:
            self.source = source
            self.destination = destination
            self.path = self.algorithm_selector()
        else:
            if source not in self.graph:
                print("\nnie ma takiego wierzchołka startowego! Wybierz któryś z listy :")
                print(self.graph.get_all_list_of_vertexes())
            if destination not in self.graph:
                print("\nnie ma takiego wierzchołka docelowego! Wybierz któryś z listy :")
                print(self.graph.get_all_list_of_vertexes())

    def dijkstra(self):
        visited = list()
        tablica_kosztow = {self.source: 0, self.destination: float('inf')}
        tablica_rodzicow = {self.destination: None}

        for sasiad in self.graph.adjacencies[self.source]:
            tablica_rodzicow[sasiad.destination] = self.source
        v = self.source

        while v:
            c = tablica_kosztow[v]
            for edge in self.graph.adjacencies[v]:
                nc = c + edge.weight
                if edge.destination not in tablica_kosztow.keys():
                    tablica_kosztow[edge.destination] = nc
                    tablica_rodzicow[edge.destination] = v
                elif tablica_kosztow[edge.destination] > nc:
                    tablica_kosztow[edge.destination] = nc
                    tablica_rodzicow[edge.destination] = v
            visited.append(v)
            if v != self.destination:
                v = self.najtanszy_wierzcholek(tablica_kosztow, visited)
            else:
                v = None

        v = self.destination
        wynik = list()

        while v in tablica_rodzicow.keys():
            wynik.append(v)
            v = tablica_rodzicow[v]
        wynik.append(self.source)
        wynik.reverse()
        # self.price = price[self.destination]

        return wynik

    # def dijkstra_project(self, wertex: Vertex):
    #     result = dict()
    #     result[wertex] = list()
    #     visited = list()
    #     price = {self.source: 0, self.destination: float('inf')}
    #     parents = {self.destination: None}
    #     for neighbor in self.graph.adjacencies[self.source]:
    #         parents[neighbor.destination] = self.source
    #     v = self.source
    #     while v:
    #         c = price[v]
    #         for edge in self.graph.adjacencies[v]:
    #             nc = c + edge.weight
    #             if edge.destination not in price.keys():
    #                 price[edge.destination] = nc
    #                 parents[edge.destination] = v
    #             elif price[edge.destination] > nc:
    #                 price[edge.destination] = nc
    #                 parents[edge.destination] = v
    #         visited.append(v)
    #         if v != self.destination:
    #             v = self.lowVert(price, visited)
    #             # result[wertex].append(v)
    #         else:
    #             v = None
    #     v = self.destination
    #     while v in parents.keys():
    #         result[wertex].append(v)
    #         v = parents[v]
    #     result[wertex].append(self.source)
    #     # result.__reversed__()
    #     self.price = price[self.destination]
    #     return result

    def najtanszy_wierzcholek(self, tablica_kosztow, visited):
        lc = float('inf')  # infinity
        for vertex in tablica_kosztow.keys():
            if tablica_kosztow[vertex] < lc and vertex not in visited:
                lc = tablica_kosztow[vertex]
                wynik = vertex
        return wynik

    def bfs(self):
        visited = [self.source]
        queue1 = Queue()
        queue1.enqueue([self.source])

        while queue1:
            p = queue1.dequeue()
            v = p[-1]
            for n in self.graph.adjacencies[v]:
                if n.destination not in visited:
                    np = p.copy()
                    np.append(n.destination)
                    visited.append(n.destination)
                    queue1.enqueue(np)
                    if n.destination == self.destination:
                        while queue1:
                            wynik = queue1.dequeue()
                            if wynik[0] == self.source and wynik[-1] == self.destination:
                                break
        return wynik

    def algorithm_selector(self):
        if self.graph.adjacencies[self.source][0].weight is not None:
            print("Szukam najbliższych ścieżek przy pomocy algorytmu Dijkstry...")
            # self.wazony = True
            # for wierzcholki in self.graph.get_all_list_of_vertexes():
            #     self.dijkstra()
            return self.dijkstra()
        else:
            print("Szukam najbliższych ścieżek przy pomocy algorytmu BFS...")
            # self.wazony = False
            return self.bfs()

    def show(self):
        wynik = dict()
        krawedzie = list()
        path = self.path.copy()
        while len(path) > 1:
            v = path.pop(0) # wierzcholek poczatkowy , a path[0] to wierzcholek koncowy
            krawedzie.append((v, path[0]))

        wynik[path[0]] = krawedzie
        print(wynik)
        self.graph.show(krawedzie)


# ------------------------------------------ PROJECT  ------------------------------------------- #


def all_shortest_paths(graph: Graph, start: Any):
    # creating all paths and showing them
    for x2 in graph.get_all_list_of_vertexes():
        graphpath2 = GraphPath(graph, start, x2)
        graphpath2.show()


graph2 = Graph()
graph2.create_vertex("0")
graph2.create_vertex("1")
graph2.create_vertex("2")
graph2.create_vertex("3")
graph2.create_vertex("4")
graph2.create_vertex("5")
graph2.create_vertex("6")

graph2.add(EdgeType(2), graph2.get_vertex("0"), graph2.get_vertex("1"), 5)
graph2.add(EdgeType(2), graph2.get_vertex("0"), graph2.get_vertex("4"), 4)
graph2.add(EdgeType(2), graph2.get_vertex("0"), graph2.get_vertex("2"), 313)
graph2.add(EdgeType(2), graph2.get_vertex("0"), graph2.get_vertex("5"), 11)
graph2.add(EdgeType(2), graph2.get_vertex("2"), graph2.get_vertex("4"), 6)
graph2.add(EdgeType(2), graph2.get_vertex("2"), graph2.get_vertex("6"), 10)
graph2.add(EdgeType(2), graph2.get_vertex("2"), graph2.get_vertex("3"), 7)
graph2.add(EdgeType(2), graph2.get_vertex("2"), graph2.get_vertex("5"), 4)
graph2.add(EdgeType(2), graph2.get_vertex("3"), graph2.get_vertex("5"), 9)
graph2.add(EdgeType(2), graph2.get_vertex("3"), graph2.get_vertex("6"), 7)


graph3 = Graph()
graph3.create_vertex("0")
graph3.create_vertex("1")
graph3.create_vertex("2")
graph3.create_vertex("3")
graph3.create_vertex("4")

graph3.add(EdgeType(2), graph3.get_vertex("0"), graph3.get_vertex("1"), 5)
graph3.add(EdgeType(2), graph3.get_vertex("1"), graph3.get_vertex("2"), 6)
graph3.add(EdgeType(2), graph3.get_vertex("1"), graph3.get_vertex("3"), 3)
graph3.add(EdgeType(2), graph3.get_vertex("1"), graph3.get_vertex("4"), 8)
graph3.add(EdgeType(2), graph3.get_vertex("2"), graph3.get_vertex("3"), 4)
graph3.add(EdgeType(2), graph3.get_vertex("2"), graph3.get_vertex("4"), 5)
graph3.add(EdgeType(2), graph3.get_vertex("3"), graph3.get_vertex("0"), 7)


#     graphpath1 = GraphPath(graph2, graph2.get_vertex("0"), x)
#     graphpath1.show()


print("\n-----------------------KLASA GRAPH----------------------- : ")
print("\nDISPLAYING ALL VERTICES ( AS THEIR DATA ) : ")  # as data, because representation of it is data
print(graph2.get_all_list_of_vertexes())

print("\nDISPLAYING VERTICES WITH THEIR INDEXES AND THEIR ADJACENCY LIST")
print(graph2)

print("\nGETTING THE NEIGHBOURS OF GIVEN VERTEX IN GRAPH : ")
print(graph2.get_neighbours(graph2.get_vertex("0")))

print("\nGETTING THE ADJACENCY LIST IN GRAPH : ")
print(graph2.get_adjacency())

# USING TRAVERSE DEPTH FIRST
graph2.traverse_depth_first(visit)

print("\nPRINTING THE LIST OF ALL TRAVERSED VERTEXES :")
print(traversed_vertex_list)

# CLEARING THE LIST
traversed_vertex_list.clear()

print("\n\n-----------------------KLASA GRAPHPATH + PROJEKT----------------------- : ")

# DISPLAYING THE GRAPH
# graph2.show()

all_shortest_paths(graph2, graph2.get_vertex("0"))
# all_shortest_paths(graph1, graph1.get_vertex("2"))
# all_shortest_paths(graph3, graph3.get_vertex("0"))




