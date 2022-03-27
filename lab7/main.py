from enum import Enum
from typing import Optional, Callable
from typing import Dict
from typing import List
from typing import Any
from graphviz import Digraph


class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None


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
                temp_queue = temp_queue + str(self._storage.node(i).data) # dla ostatniego elementu - brak przecinka
            else:
                temp_queue = temp_queue + str(self._storage.node(i).data) + ", "
        return temp_queue


class EdgeType(Enum):
    directed = 1
    undirected = 2


class Vertex:
    data: Any
    index: int

    def __init__(self, data1, index1):
        self.data = data1
        self.index = index1


class Edge:
    source: Vertex
    destination: Vertex
    weight: Optional[float]

    def __init__(self, source1, destination1, weight1):
        self.source = source1
        self.destination = destination1
        self.weight = weight1

    def __repr__(self):
        return "{}: v{}".format(self.destination.data, self.destination.index)


class Graph:
    adjacencies: Dict[Vertex, List[Edge]]

    def __init__(self):
        self.adjacencies = dict()  # słownik

    def create_vertex(self, data: Any):
        self.adjacencies[Vertex(data, len(self.adjacencies))] = list()  # left side is key ( vertex ) while its value is adjacency list that is list()
        # # adding key
        # self.adjacencies.??add??(Vertex(data, len(self.adjacencies)))
        # # adding value to the key
        # self.adjacencies[Vertex(data, len(self.adjacencies))] = list()
        # return Vertex(data, len(self.adjacencies))

    def get_all_list_of_vertexes(self) -> List[Vertex]:
        graph_vertices_keys1 = [x for x in self.adjacencies.keys()]
        return graph_vertices_keys1

    def get_all_neighbours(self) -> List[List[Edge]]:
        graph_vertices_keys1 = [x for x in self.adjacencies.values()]
        return graph_vertices_keys1

    def get_neighbours(self, key_temp) -> List[Edge]:
        return self.adjacencies.get(key_temp)

    def get_adjacency(self):
        return self.adjacencies

    def add_directed_edge(self, source: Vertex, destination: Vertex, weight: Optional[float] = None) -> None:
        if source not in self.adjacencies:
            self.adjacencies[Vertex(source.data, len(self.adjacencies))] = list()
        if destination not in self.adjacencies:
            self.adjacencies[Vertex(destination.data, len(self.adjacencies))] = list()
        self.adjacencies[source].append(Edge(source, destination, weight))

    def add_undirected_edge(self, source: Vertex, destination: Vertex, weight: Optional[float] = None) -> None:
        if source not in self.adjacencies:
            self.adjacencies[Vertex(source.data, len(self.adjacencies))] = list()
        if destination not in self.adjacencies:
            self.adjacencies[Vertex(destination.data, len(self.adjacencies))] = list()
        self.adjacencies[source].append(Edge(source, destination, weight))
        self.adjacencies[destination].append(Edge(destination, source, weight))

    def add(self, edge: EdgeType, source: Vertex, destination: Vertex, weight: Optional[float] = None):
        # it's enum, we must find the name, not the value
        if edge.name == "directed":
            self.add_directed_edge(source, destination, weight)
        elif edge.name == "undirected":
            self.add_undirected_edge(source, destination, weight)
        else:
            raise ValueError("WRONG EDGE TYPE ! WRITE EdgeType(1) OR EdgeType(2) to work !! \n")

    def traverse_bfs(self, visit: Callable[[Any], None]):
        vertex = list(self.adjacencies.keys())[0]  # rzutowanie do listy
        visited = []
        queue1 = Queue()
        queue1.enqueue(vertex)
        while len(queue1) != 0:
            v = queue1.dequeue()
            visit(v)
            for x in self.adjacencies[v]:
                if x.destination not in visited:
                    visited.append(x.destination)
                    queue1.enqueue(x.destination)
                else:
                    return

    def dfs(self, vertex: Vertex, visited: List[Vertex], visit):
        visit(vertex)
        visited.append(vertex)
        for x in self.adjacencies[vertex]:  # edgez
            if x.destination not in visited:
                self.dfs(x.destination, visited, visit)

    def traverse_depth_first(self, visit):
        vertex = list(self.adjacencies.keys())[0]
        visited = []
        self.dfs(vertex, visited, visit)

    def show(self):
        filename = "graph"
        graph = Digraph()
        visited = []
        graph_vertices = self.adjacencies.keys()
        for x in graph_vertices:
            self.show_support(x, graph, visited)
        # displaying the graph
        graph.render(filename=filename, directory='C:\\2022 studia\\PRZEDMIOTY\\AiSD\\LAB_IgorKawczynski\\Lab7',
                     view=True)  # problems with jpg

    def show_support(self, vertex: Vertex, graph, visited: List):
        if vertex not in visited:
            graph.node(str(vertex.index), str(vertex.data))
            visited.append(vertex)
            for x in self.adjacencies[vertex]:  # x is a vertex neighbour
                desc = ""
                if x.weight is not None:
                    desc += f"{x.weight}"
                graph.edge(str(x.source.index), str(x.destination.index), label=desc)  # label as a weight
                if not (x.destination in visited):
                    self.show_support(x.destination, graph, visited)
        else:
            return True

    def __str__(self):
        temp = ""
        for vertexes in self.adjacencies:
            temp = temp + "{}: v{} ----> {} \n".format(vertexes.data, vertexes.data, self.adjacencies[vertexes])
        return temp


traversed_vertex_list = []


def __visit(vertex):
    traversed_vertex_list.append(vertex.data)


# CREATING THE GRAPH
graph1 = Graph()
graph1.create_vertex(0)
graph1.create_vertex(1)
graph1.create_vertex(2)
graph1.create_vertex(3)
graph1.create_vertex(4)
graph1.create_vertex(5)

# CREATING THE LIST OF VERTICES, WITH KEYS
graph_vertices_keys = [x for x in graph1.adjacencies.keys()]

# ADDING EDGES, ALL DIRECTED, WITH WEIGHS
graph1.add(EdgeType(1), graph_vertices_keys[0], graph_vertices_keys[1], weight=1)
graph1.add(EdgeType(1), graph_vertices_keys[0], graph_vertices_keys[5], weight=3)
graph1.add(EdgeType(1), graph_vertices_keys[2], graph_vertices_keys[1], weight=2)
graph1.add(EdgeType(1), graph_vertices_keys[2], graph_vertices_keys[3], weight=5)
graph1.add(EdgeType(1), graph_vertices_keys[3], graph_vertices_keys[4], weight=2)
graph1.add(EdgeType(1), graph_vertices_keys[4], graph_vertices_keys[1], weight=4)
graph1.add(EdgeType(1), graph_vertices_keys[4], graph_vertices_keys[5], weight=4)
graph1.add(EdgeType(1), graph_vertices_keys[5], graph_vertices_keys[1], weight=1)
graph1.add(EdgeType(1), graph_vertices_keys[5], graph_vertices_keys[2], weight=2)

# DISPLAYING ALL VERTICES ( AS OBJECTS - THEIR DATA - THEIR INDEXES )
for x in graph1.adjacencies.keys():
    print(x)
    print(x.data)
    print(x.index)

# DISPLAYING VERTICES WITH THEIR INDEXES AND THEIR ADJACENCY LIST
print(graph1)

# GETTING ALL OF THE VERTEXES IN GRAPH
print(graph1.get_all_list_of_vertexes())

# GETTING ALL OF THE NEIGHBOURS IN GRAPH
print(graph1.get_all_neighbours())

# GETTING THE NEIGHBOURS OF GIVEN VERTEX IN GRAPH
print(graph1.get_neighbours(graph_vertices_keys[0]))

# GETTING THE ADJACENCY LIST IN GRAPH
print(graph1.get_adjacency())

# USING TRAVERSE DEPTH FIRST
graph1.traverse_depth_first(__visit)
# PRINTING THE LIST OF ALL TRAVERSED VERTEXES
print(traversed_vertex_list)
# CLEARING THE LIST
traversed_vertex_list.clear()

# DISPLAYING THE GRAPH IN JPG
graph1.show()
