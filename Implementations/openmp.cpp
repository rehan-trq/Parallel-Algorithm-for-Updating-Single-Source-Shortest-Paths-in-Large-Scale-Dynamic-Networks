#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
#include <limits>
#include <omp.h>
#include <algorithm>

using namespace std;

const int INF = 1e9; // Avoid numeric_limits<int>::max()

struct Edge {
    int dest;
    int weight;
};

struct GraphChange {
    int u, v;
    bool isInsertion;
    int weight;
};

class ParallelDynamicSSSP {
private:
    int numVertices;
    vector<vector<Edge>> graph;
    vector<int> dist;
    vector<int> parent;
    vector<int> affected;       // Use int instead of bool
    vector<int> affectedDel;

public:
    ParallelDynamicSSSP(int n, int threads = 4) {
        numVertices = n;
        omp_set_num_threads(threads);
        graph.resize(n);
        dist.assign(n, INF);
        parent.assign(n, -1);
        affected.assign(n, 0);
        affectedDel.assign(n, 0);
    }

    void addEdge(int u, int v, int w) {
        if (u >= numVertices || v >= numVertices || u < 0 || v < 0) return;
        graph[u].push_back({v, w});
        graph[v].push_back({u, w});
    }

    void removeEdge(int u, int v) {
        auto& edgesU = graph[u];
        edgesU.erase(remove_if(edgesU.begin(), edgesU.end(),
                      [&](const Edge& e) { return e.dest == v; }),
                     edgesU.end());

        auto& edgesV = graph[v];
        edgesV.erase(remove_if(edgesV.begin(), edgesV.end(),
                      [&](const Edge& e) { return e.dest == u; }),
                     edgesV.end());
    }

    void computeInitialSSSP(int source) {
        fill(dist.begin(), dist.end(), INF);
        fill(parent.begin(), parent.end(), -1);

        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>>
            pq;
        dist[source] = 0;
        pq.push({0, source});

        while (!pq.empty()) {
            int d_u = pq.top().first;
            int u = pq.top().second;
            pq.pop();

            if (d_u > dist[u]) continue;

            for (const Edge& edge : graph[u]) {
                int v = edge.dest;
                int w = edge.weight;

                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    parent[v] = u;
                    pq.push({dist[v], v});
                }
            }
        }
    }

    void processChangesInParallel(const vector<GraphChange>& changes) {
        fill(affected.begin(), affected.end(), 0);
        fill(affectedDel.begin(), affectedDel.end(), 0);

        #pragma omp parallel
        {
            #pragma omp for nowait
            for (int i = 0; i < changes.size(); ++i) {
                const auto& chg = changes[i];
                if (!chg.isInsertion) handleDeletion(chg.u, chg.v);
            }

            #pragma omp for nowait
            for (int i = 0; i < changes.size(); ++i) {
                const auto& chg = changes[i];
                if (chg.isInsertion) handleInsertion(chg.u, chg.v, chg.weight);
            }
        }

        propagateDeletions();
        propagateUpdates();
    }

    void handleDeletion(int u, int v) {
        removeEdge(u, v);
        if (parent[v] == u || parent[u] == v) {
            int x = (parent[v] == u) ? v : u;
            dist[x] = INF;
            affectedDel[x] = 1;
            affected[x] = 1;
        }
    }

    void handleInsertion(int u, int v, int w) {
        addEdge(u, v, w);
        if (dist[u] + w < dist[v]) {
            dist[v] = dist[u] + w;
            parent[v] = u;
            affected[v] = 1;
        }
        if (dist[v] + w < dist[u]) {
            dist[u] = dist[v] + w;
            parent[u] = v;
            affected[u] = 1;
        }
    }

    void propagateDeletions() {
        bool changed;
        do {
            changed = false;

            #pragma omp parallel
            {
                bool localChanged = false;

                #pragma omp for
                for (int v = 0; v < numVertices; ++v) {
                    if (affectedDel[v]) {
                        affectedDel[v] = 0;
                        for (const Edge& edge : graph[v]) {
                            #pragma omp atomic write
                            affectedDel[edge.dest] = 1;
                            #pragma omp atomic write
                            affected[edge.dest] = 1;
                        }
                        localChanged = true;
                    }
                }

                #pragma omp atomic
                changed |= localChanged;
            }
        } while (changed);
    }

    void propagateUpdates() {
        bool updated = true;
        int iterCount = 0;
        const int MAX_ITERATIONS = 1000;

        while (updated && iterCount++ < MAX_ITERATIONS) {
            updated = false;

            #pragma omp parallel
            {
                bool localUpdated = false;

                #pragma omp for schedule(dynamic)
                for (int v = 0; v < numVertices; ++v) {
                    if (affected[v] != 0) {
                        affected[v] = 0;

                        for (const Edge& edge : graph[v]) {
                            int neighbor = edge.dest;
                            int weight = edge.weight;

                            if (dist[v] + weight < dist[neighbor]) {
                                dist[neighbor] = dist[v] + weight;
                                parent[neighbor] = v;

                                #pragma omp atomic write
                                affected[neighbor] = 1;

                                localUpdated = true;
                            }
                        }
                    }
                }

                #pragma omp atomic
                updated |= localUpdated;
            }
        }
    }

    bool validateSSSP(int source) {
        if (dist[source] != 0 || parent[source] != -1) {
            cout << "Error: Source node has incorrect values\n";
            return false;
        }

        for (int v = 0; v < numVertices; ++v) {
            if (v == source) continue;
            if (dist[v] == INF && parent[v] != -1) {
                cout << "Error: Vertex " << v << " has no distance but has parent " << parent[v] << "\n";
                return false;
            }
            if (dist[v] < INF && parent[v] == -1) {
                cout << "Error: Vertex " << v << " has distance " << dist[v] << " but no parent\n";
                return false;
            }
        }
        return true;
    }

    void printSSSP() {
        cout << "Vertex \t Distance from Source \t Parent\n";
        for (int i = 0; i < numVertices; ++i) {
            cout << i << " \t ";
            if (dist[i] == INF)
                cout << "INF \t\t\t ";
            else
                cout << dist[i] << " \t\t\t ";
            cout << parent[i] << "\n";
        }
    }

    int countReachable() {
        int count = 0;
        for (int d : dist) if (d < INF) ++count;
        return count;
    }

    double getAverageDistance() {
        long long sum = 0;
        int count = 0;
        for (int d : dist) {
            if (d < INF) {
                sum += d;
                ++count;
            }
        }
        return count > 0 ? static_cast<double>(sum) / count : 0;
    }
};

int main() {
    int numNodes = 11;
    ParallelDynamicSSSP sssp(numNodes);

    // Original edges
    sssp.addEdge(1, 2, 5);
    sssp.addEdge(1, 3, 3);
    sssp.addEdge(1, 4, 7);
    sssp.addEdge(2, 5, 2);
    sssp.addEdge(2, 6, 4);
    sssp.addEdge(3, 6, 6);
    sssp.addEdge(3, 7, 1);
    sssp.addEdge(4, 7, 8);
    sssp.addEdge(5, 8, 3);
    sssp.addEdge(6, 8, 5);
    sssp.addEdge(6, 9, 2);
    sssp.addEdge(7, 9, 4);
    sssp.addEdge(8, 10, 7);
    sssp.addEdge(9, 10, 6);

    const int SOURCE_VERTEX = 0;

    cout << "Computing SSSP from source vertex " << SOURCE_VERTEX << "..." << endl;
    double startTime = omp_get_wtime();
    sssp.computeInitialSSSP(SOURCE_VERTEX);
    double endTime = omp_get_wtime();

    cout << "Initial SSSP computation time: " << (endTime - startTime) << " seconds" << endl;
    if (sssp.validateSSSP(SOURCE_VERTEX))
        cout << "Initial SSSP tree is valid.\n";
    else
        cout << "Initial SSSP tree validation failed!\n";

    cout << "Reachable vertices: " << sssp.countReachable() << " out of " << numNodes << endl;
    cout << "Average distance: " << sssp.getAverageDistance() << endl;
    sssp.printSSSP();

    vector<GraphChange> changes = {
        {1, 4, false, 7}, {7, 9, false, 4}, {3, 7, false, 1},
        {9, 10, false, 6}, {8, 10, false, 7}, {1, 3, false, 3},
        {9, 4, true, 4}, {7, 1, true, 9}, {9, 4, true, 1},
        {1, 7, true, 4}, {3, 5, true, 3}, {1, 5, true, 6},
        {4, 10, true, 6}, {6, 4, true, 10}, {0, 8, true, 3}, {10, 3, true, 3}
    };

    cout << "\nSimulating changes to the graph...\n";
    for (const auto& chg : changes) {
        if (chg.isInsertion)
            cout << "Insert edge: " << chg.u << " -> " << chg.v << " (weight: " << chg.weight << ")\n";
        else
            cout << "Delete edge: " << chg.u << " -> " << chg.v << " (weight: " << chg.weight << ")\n";
    }

    cout << "\nProcessing changes...\n";
    startTime = omp_get_wtime();
    sssp.processChangesInParallel(changes);
    endTime = omp_get_wtime();

    cout << "SSSP update time: " << (endTime - startTime) << " seconds\n";
    if (sssp.validateSSSP(SOURCE_VERTEX))
        cout << "Updated SSSP tree is valid.\n";
    else
        cout << "Warning: Updated SSSP tree validation failed!\n";

    cout << "Updated reachable vertices: " << sssp.countReachable() << " out of " << numNodes << endl;
    cout << "Updated average distance: " << sssp.getAverageDistance() << endl;

    cout << "\nSSSP Tree after changes:\n";
    sssp.printSSSP();

    return 0;
}