#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <limits>
#include <ctime>
#include <omp.h>
#include <mpi.h>
#include <metis.h>

using std::vector;
using std::pair;
using std::string;

struct Logger {
    int rank;
    bool enabled;

    void log(const string& component, const string& message) {
        if (!enabled) return;
        time_t now = time(nullptr);
        char timeStr[20];
        strftime(timeStr, sizeof(timeStr), "%H:%M:%S", localtime(&now));
        std::cout << "[" << timeStr << "] Rank " << rank 
                  << " [" << component << "]: " << message << std::endl;
    }
};

struct Connection {
    int from, to;
    int cost;
    Connection(int f = 0, int t = 0, int c = 1) : from(f), to(t), cost(c) {}
};

class DistributedSSP {
private:
    int nodeCount;
    vector<vector<pair<int, int>>> adjacency;
    vector<int> distances;
    vector<int> predecessors;
    vector<bool> modified;
    vector<int> partitionMap;
    Logger logger;
    int threadCount;
    int selfRank;
    int totalRanks;

    int getHomeRank(int node) {
        return partitionMap[node];
    }

    void distributeUpdate(int node, int newDist, int newParent) {
        int buffer[3] = {node, newDist, newParent};
        for (int destRank = 0; destRank < totalRanks; destRank++) {
            if (destRank != selfRank) {
                MPI_Send(buffer, 3, MPI_INT, destRank, 0, MPI_COMM_WORLD);
            }
        }
    }

    void handleIncoming() {
        MPI_Status status;
        int flag;
        while (true) {
            MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
            if (!flag) break;
            
            int buffer[3];
            MPI_Recv(buffer, 3, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &status);
            
            int node = buffer[0];
            int newDist = buffer[1];
            int newParent = buffer[2];
            
            if (newDist < distances[node]) {
                #pragma omp critical(update_section)
                {
                    if (newDist < distances[node]) {
                        distances[node] = newDist;
                        predecessors[node] = newParent;
                        modified[node] = true;
                    }
                }
            }
        }
    }

public:
    DistributedSSP(int nodes, bool verbose, int threads) 
        : nodeCount(nodes), logger{0, verbose} {
        
        adjacency.resize(nodes);
        distances.resize(nodes, std::numeric_limits<int>::max());
        predecessors.resize(nodes, -1);
        modified.resize(nodes, false);
        partitionMap.resize(nodes);
        
        MPI_Comm_rank(MPI_COMM_WORLD, &selfRank);
        MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);
        logger.rank = selfRank;
        
        omp_set_num_threads(threads);
    }

    void addConnection(int from, int to, int weight) {
        if (from >= 0 && from < nodeCount && to >= 0 && to < nodeCount) {
            adjacency[from].emplace_back(to, weight);
            adjacency[to].emplace_back(from, weight);
            logger.log("Network", "Added connection: " + std::to_string(from) + "-" + 
                      std::to_string(to) + " (cost: " + std::to_string(weight) + ")");
        }
    }

    void removeConnection(int from, int to) {
        if (from >= 0 && from < nodeCount && to >= 0 && to < nodeCount) {
            auto removeEdge = [&](vector<pair<int, int>>& edges) {
                edges.erase(std::remove_if(edges.begin(), edges.end(),
                    [to](const pair<int, int>& p) { return p.first == to; }), edges.end());
            };
            
            removeEdge(adjacency[from]);
            removeEdge(adjacency[to]);
            logger.log("Network", "Removed connection: " + std::to_string(from) + "-" + 
                      std::to_string(to));
        }
    }

    void partitionNetwork() {
        if (selfRank == 0) {
            logger.log("Partition", "Starting METIS-based partitioning");
        }
        
        idx_t nodes = nodeCount;
        idx_t constraints = 1;
        vector<idx_t> xadj(nodes + 1, 0), adjncy, adjwgt;
        
        // Build adjacency structure for METIS
        for (int i = 0; i < nodes; i++) {
            xadj[i+1] = xadj[i] + adjacency[i].size();
        }
        
        adjncy.resize(xadj[nodes]);
        adjwgt.resize(xadj[nodes]);
        
        for (int i = 0; i < nodes; i++) {
            int offset = xadj[i];
            for (size_t j = 0; j < adjacency[i].size(); j++) {
                adjncy[offset + j] = adjacency[i][j].first;
                adjwgt[offset + j] = adjacency[i][j].second;
            }
        }
        
        idx_t parts = totalRanks;
        idx_t edgeCut;
        idx_t options[METIS_NOPTIONS];
        METIS_SetDefaultOptions(options);
        options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
        options[METIS_OPTION_NUMBERING] = 0;
        
        vector<idx_t> assignment(nodes);
        int result = METIS_PartGraphKway(&nodes, &constraints, xadj.data(), adjncy.data(),
                                        NULL, NULL, adjwgt.data(), &parts, NULL,
                                        NULL, options, &edgeCut, assignment.data());
        
        if (result != METIS_OK) {
            logger.log("Error", "METIS partitioning failed");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        if (selfRank == 0) {
            logger.log("Partition", "Edge cuts: " + std::to_string(edgeCut));
        }
        
        MPI_Bcast(assignment.data(), nodes, MPI_INT, 0, MPI_COMM_WORLD);
        
        for (int i = 0; i < nodes; i++) {
            partitionMap[i] = assignment[i];
        }
    }

    void calculateShortestPaths(int origin) {
        std::fill(distances.begin(), distances.end(), std::numeric_limits<int>::max());
        std::fill(predecessors.begin(), predecessors.end(), -1);
        std::fill(modified.begin(), modified.end(), false);
        
        distances[origin] = 0;
        bool complete = false;
        int iteration = 0;
        
        if (selfRank == 0) {
            time_t now = time(nullptr);
            char timeStr[20];
            strftime(timeStr, sizeof(timeStr), "%H:%M:%S", localtime(&now));
            logger.log("Execution", "Started at " + string(timeStr));
        }
        
        while (!complete) {
            int minDistance = std::numeric_limits<int>::max();
            int closestNode = -1;
            
            for (int node = 0; node < nodeCount; node++) {
                if (!modified[node] && distances[node] < minDistance) {
                    minDistance = distances[node];
                    closestNode = node;
                }
            }
            
            struct { int dist; int node; } local = {minDistance, closestNode}, global;
            MPI_Allreduce(&local, &global, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
            
            if (global.dist == std::numeric_limits<int>::max()) {
                complete = true;
                continue;
            }
            
            int current = global.node;
            modified[current] = true;
            
            vector<pair<int, int>> updates;
            if (current >= 0 && current < nodeCount) {
                for (auto& edge : adjacency[current]) {
                    int neighbor = edge.first;
                    int weight = edge.second;
                    if (distances[current] != std::numeric_limits<int>::max() &&
                        distances[current] + weight < distances[neighbor]) {
                        updates.emplace_back(neighbor, distances[current] + weight);
                    }
                }
            }
            
            int updateCount = updates.size();
            vector<int> allCounts(totalRanks);
            MPI_Allgather(&updateCount, 1, MPI_INT, allCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
            
            for (int target = 0; target < totalRanks; target++) {
                if (allCounts[target] > 0) {
                    vector<pair<int, int>> targetUpdates;
                    if (target == selfRank) {
                        targetUpdates = updates;
                    } else {
                        targetUpdates.resize(allCounts[target]);
                    }
                    
                    MPI_Datatype pairType;
                    MPI_Type_contiguous(2 * sizeof(int), MPI_BYTE, &pairType);
                    MPI_Type_commit(&pairType);
                    MPI_Bcast(targetUpdates.data(), allCounts[target], pairType, target, MPI_COMM_WORLD);
                    MPI_Type_free(&pairType);
                    
                    for (auto& update : targetUpdates) {
                        int node = update.first;
                        int newDist = update.second;
                        if (newDist < distances[node]) {
                            distances[node] = newDist;
                            predecessors[node] = current;
                        }
                    }
                }
            }
            
            iteration++;
            MPI_Barrier(MPI_COMM_WORLD);
        }
        
        if (selfRank == 0) {
            double duration = MPI_Wtime();
            logger.log("Execution", "Completed in " + std::to_string(iteration) + " iterations");
            logger.log("Timing", "Total execution time: " + std::to_string(duration) + "s");
        }
    }

    void showStatistics() {
        vector<int> finalDist(nodeCount);
        MPI_Allreduce(distances.data(), finalDist.data(), nodeCount, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        
        if (selfRank == 0) {
            std::cout << "\n[Analysis] Path Statistics:" << std::endl;
            int reachable = 0;
            long long total = 0;
            
            for (int i = 0; i < nodeCount; i++) {
                if (finalDist[i] != std::numeric_limits<int>::max()) {
                    reachable++;
                    total += finalDist[i];
                }
            }
            
            std::cout << "Reachable nodes: " << reachable << "/" << nodeCount 
                      << " (" << (reachable * 100.0 / nodeCount) << "%)" << std::endl;
                      
            if (reachable > 0) {
                std::cout << "Average distance: " << (double)total / reachable << std::endl;
            }
        }
    }
};

int main(int argc, char* argv[]) {
    int threadSupport;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threadSupport);
    
    if (threadSupport < MPI_THREAD_MULTIPLE) {
        std::cerr << "[Failure] MPI requires full thread support" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 2 && rank == 0) {
        std::cerr << "Usage: " << argv[0] << " <input_file> [source] [verbose] [threads]" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    string inputFile = (argc > 1) ? argv[1] : "as20000102.txt";
    int sourceNode = (argc > 2) ? std::stoi(argv[2]) : 0;
    bool verboseMode = (argc > 3) ? (std::stoi(argv[3]) != 0) : false;
    int numThreads = (argc > 4) ? std::stoi(argv[4]) : omp_get_max_threads();
    
    if (rank == 0) {
        std::cout << "[System] Distributed Shortest Path Solver" << std::endl;
        std::cout << "Dataset: " << inputFile << std::endl;
        std::cout << "Source: " << sourceNode << std::endl;
        std::cout << "Verbosity: " << (verboseMode ? "Active" : "Silent") << std::endl;
        std::cout << "Threads: " << numThreads << std::endl;
    }
    
    int nodes = 0;
    vector<Connection> connections;
    
    if (rank == 0) {
        std::ifstream reader(inputFile);
        if (!reader.is_open()) {
            std::cerr << "[ERROR] Failed to open input file: " << inputFile << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        string line;
        while (getline(reader, line)) {
            if (line.empty() || line[0] == '#') continue;
            int from, to, weight;
            std::istringstream iss(line);
            if (iss >> from >> to >> weight) {
                nodes = std::max(nodes, std::max(from, to) + 1);
                connections.emplace_back(from, to, weight);
            }
        }
        reader.close();

        std::cout << "[Data] Nodes: " << nodes << ", Connections: " << connections.size() << std::endl;

        if (nodes == 0) {
            std::cerr << "[ERROR] No valid data parsed from file!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    DistributedSSP pathFinder(nodes, verboseMode, numThreads);

    int connectionCount = connections.size();
    MPI_Bcast(&connectionCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        connections.resize(connectionCount);
    }

    MPI_Datatype connectionType;
    MPI_Type_contiguous(3, MPI_INT, &connectionType);
    MPI_Type_commit(&connectionType);
    
    MPI_Bcast(connections.data(), connectionCount, connectionType, 0, MPI_COMM_WORLD);
    MPI_Type_free(&connectionType);

    pathFinder.partitionNetwork();

    int perProcess = connectionCount / size;
    int start = rank * perProcess;
    int end = (rank == size - 1) ? connectionCount : start + perProcess;

    for (int i = start; i < end; i++) {
        const auto& conn = connections[i];
        pathFinder.addConnection(conn.from, conn.to, conn.cost);
    }

    double startTime = MPI_Wtime();
    pathFinder.calculateShortestPaths(sourceNode);
    double endTime = MPI_Wtime();

    if (rank == 0) {
        std::cout << "Processing completed in " << (endTime - startTime) << " seconds" << std::endl;
    }

    pathFinder.showStatistics();

    MPI_Finalize();
    return 0;
}
