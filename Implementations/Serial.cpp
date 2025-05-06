#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>

using namespace std;

// Structure to represent an edge in the graph
struct Edge {
    int dest;
    int weight;
};

// Structure to represent a change in the graph
struct Change {
    int src, dest;
    bool isInsert; // true for insertion, false for deletion
    int weight;    // Weight of the edge
};

// Function to perform Dijkstra's algorithm
void dijkstra(int source, const unordered_map<int, vector<Edge>>& graph, int numVertices, vector<int>& distance, vector<int>& parent) {
    // Distance vector initialized to infinity
    fill(distance.begin(), distance.end(), numeric_limits<int>::max());
    // Parent vector initialized to -1
    fill(parent.begin(), parent.end(), -1);

    // Min-heap priority queue to store (distance, vertex)
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;

    // Initialize the source vertex
    distance[source] = 0;
    pq.push({0, source});

    while (!pq.empty()) {
        int currentDistance = pq.top().first;
        int currentVertex = pq.top().second;
        pq.pop();

        // Skip if the current distance is not optimal
        if (currentDistance > distance[currentVertex]) {
            continue;
        }

        // Explore neighbors
        if (graph.find(currentVertex) != graph.end()) {
            for (const Edge& edge : graph.at(currentVertex)) {
                int neighbor = edge.dest;
                int weight = edge.weight;

                // Relaxation step
                if (distance[currentVertex] + weight < distance[neighbor]) {
                    distance[neighbor] = distance[currentVertex] + weight;
                    parent[neighbor] = currentVertex;
                    pq.push({distance[neighbor], neighbor});
                }
            }
        }
    }
}

// Function to process edge deletions
void processEdgeDeletion(int u, int v, vector<int>& distance, vector<int>& parent, vector<bool>& affectedDel) {
    if (parent[v] == u || parent[u] == v) {
        int affectedVertex = (parent[v] == u) ? v : u;
        distance[affectedVertex] = numeric_limits<int>::max();
        parent[affectedVertex] = -1;
        affectedDel[affectedVertex] = true;
    }
}

// Function to process edge insertions
void processEdgeInsertion(int u, int v, int weight, vector<int>& distance, vector<int>& parent, vector<bool>& affected) {
    if (distance[u] + weight < distance[v]) {
        distance[v] = distance[u] + weight;
        parent[v] = u;
        affected[v] = true;
    }
    if (distance[v] + weight < distance[u]) {
        distance[u] = distance[v] + weight;
        parent[u] = v;
        affected[u] = true;
    }
}

// Function to process changes in the graph
void processChanges(unordered_map<int, vector<Edge>>& graph, const vector<Change>& changes, vector<int>& distance, vector<int>& parent) {
    vector<bool> affected(distance.size(), false);
    vector<bool> affectedDel(distance.size(), false);

    // Process deletions
    for (const auto& change : changes) {
        if (!change.isInsert) {
            processEdgeDeletion(change.src, change.dest, distance, parent, affectedDel);
        }
    }

    // Process insertions
    for (const auto& change : changes) {
        if (change.isInsert) {
            processEdgeInsertion(change.src, change.dest, change.weight, distance, parent, affected);
        }
    }

    // Iteratively update affected vertices
    bool updated;
    do {
        updated = false;
        for (size_t v = 0; v < distance.size(); ++v) {
            if (affected[v] || affectedDel[v]) {
                affected[v] = affectedDel[v] = false;
                for (const auto& edge : graph[v]) {
                    if (distance[v] + edge.weight < distance[edge.dest]) {
                        distance[edge.dest] = distance[v] + edge.weight;
                        parent[edge.dest] = v;
                        updated = true;
                    }
                }
            }
        }
    } while (updated);
}

// Function to print the SSSP tree
void printSSSP(const vector<int>& distance, const vector<int>& parent) {
    cout << "Vertex \t Distance from Source \t Parent\n";
    for (size_t i = 0; i < distance.size(); ++i) {
        cout << i << " \t ";
        if (distance[i] == numeric_limits<int>::max()) {
            cout << "INF \t\t\t ";
        } else {
            cout << distance[i] << " \t\t\t ";
        }
        cout << parent[i] << "\n";
    }
}

int main() {
    // File containing the dataset
    string filename = "dataset.txt";

    // Graph representation as an adjacency list
    unordered_map<int, vector<Edge>> graph;

    // Read the dataset from the file
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Unable to open file " << filename << endl;
        return 1;
    }

    string line;
    int maxVertex = 0;

    while (getline(file, line)) {
        stringstream ss(line);
        int src, dest, weight;
        ss >> src >> dest >> weight;

        // Add the edge to the graph (both directions for undirected graph)
        graph[src].push_back({dest, weight});
        graph[dest].push_back({src, weight});

        // Track the maximum vertex index
        maxVertex = max(maxVertex, max(src, dest));
    }

    file.close();

    // Number of vertices in the graph
    int numVertices = maxVertex + 1;

    // Source vertex
    int source = 0;

    // Distance and parent vectors
    vector<int> distance(numVertices);
    vector<int> parent(numVertices);

    // Compute initial SSSP
    dijkstra(source, graph, numVertices, distance, parent);

    // Print the initial SSSP tree
    cout << "Initial SSSP Tree:\n";
    printSSSP(distance, parent);

    // Define changes (insertions and deletions)
    vector<Change> changes = {
        {2, 5, false, 8}, // Delete edge (4725, 9356)
        {4732, 10019, true, 5}, // Insert edge (4732, 10019) with weight 5
        {7518, 9354, true, 7}   // Insert edge (7518, 9354) with weight 7
    };

    // Process changes
    processChanges(graph, changes, distance, parent);

    // Print the SSSP tree after changes
    cout << "\nSSSP Tree after changes:\n";
    printSSSP(distance, parent);

    return 0;
}