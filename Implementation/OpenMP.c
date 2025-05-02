#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <float.h>
#include <omp.h>
#include <metis.h>
#include <limits.h>
#include <math.h>
#include <assert.h>

#define INF DBL_MAX

// Priority Queue Node
typedef struct {
    idx_t vertex;
    double dist;
} PQNode;

// Priority Queue (min-heap)
typedef struct {
    PQNode *nodes;
    idx_t *positions;
    idx_t size;
    idx_t capacity;
} PriorityQueue;

// Graph structure in CSR format
typedef struct {
    idx_t num_vertices;
    idx_t num_edges;
    idx_t *xadj;
    idx_t *adjncy;
    double *weights;
} Graph;

// Edge change structure
typedef struct {
    idx_t u, v;
    double weight;
    bool is_deleted;
} EdgeChange;

// Function prototypes
void read_edge_list(const char *filename, idx_t *num_vertices, idx_t *num_edges, idx_t **xadj, idx_t **adjncy, double **weights);
void partition_graph(idx_t num_vertices, idx_t *xadj, idx_t *adjncy, int num_parts, idx_t *part);
PriorityQueue* pq_create(idx_t capacity);
void pq_insert(PriorityQueue *pq, idx_t vertex, double dist);
idx_t pq_extract_min(PriorityQueue *pq, double *dist);
void compute_sssp(Graph *graph, idx_t source, double *dist, idx_t *parent, int num_threads);
void identify_affected(Graph *graph, EdgeChange *changes, int num_changes, double *dist, idx_t *parent,
                      bool *affected, bool *affected_del, int num_threads);
void update_subgraph(Graph *graph, double *dist, idx_t *parent, bool *affected, bool *affected_del,
                     int num_threads);
void free_graph(Graph *graph);
void free_pq(PriorityQueue *pq);

// Main function
int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <num_threads>\n", argv[0]);
        return 1;
    }
    int num_threads = atoi(argv[1]);

    idx_t num_vertices, num_edges;
    idx_t *xadj, *adjncy, *part;
    double *weights;

    // Load graph
    read_edge_list("/home/rehan/Downloads/BHJ-S-3-DATASET/bio-CE-CX.edges", &num_vertices, &num_edges, &xadj, &adjncy, &weights);
    printf("Total number of vertices: %d\n", num_vertices);
    printf("Total number of edges: %d\n", num_edges);
    Graph graph = {num_vertices, num_edges, xadj, adjncy, weights};

    // Partition graph
    part = malloc(num_vertices * sizeof(idx_t));
    partition_graph(num_vertices, xadj, adjncy, num_threads, part);

    // Allocate arrays
    double *dist = malloc(num_vertices * sizeof(double));
    idx_t *parent = malloc(num_vertices * sizeof(idx_t));
    bool *affected = calloc(num_vertices, sizeof(bool));
    bool *affected_del = calloc(num_vertices, sizeof(bool));

    // Initial SSSP
    double start_time = omp_get_wtime();
    compute_sssp(&graph, 0, dist, parent, num_threads);
    double end_time = omp_get_wtime();
    printf("Initial SSSP time: %f seconds\n", end_time - start_time);

    // Print initial distances (first 10 vertices)
    printf("Initial distances from source (vertex 0):\n");
    for (idx_t i = 0; i < 10 && i < num_vertices; i++) {
        double d = (dist[i] == INF) ? -1.0 : dist[i];
        printf("Vertex %d: Distance = %.6f, Parent = %d\n", i, d, parent[i]);
    }

    // Example update: delete edge 0-1 and insert edge 2-3
    EdgeChange changes[2];
    changes[0].u = 0;
    changes[0].v = 1;
    changes[0].weight = 0.0; // Not used for deletion
    changes[0].is_deleted = true;
    changes[1].u = 2;
    changes[1].v = 3;
    changes[1].weight = 1.5;
    changes[1].is_deleted = false;

    // Apply update
    start_time = omp_get_wtime();
    identify_affected(&graph, changes, 2, dist, parent, affected, affected_del, num_threads);
    update_subgraph(&graph, dist, parent, affected, affected_del, num_threads);
    end_time = omp_get_wtime();
    printf("Update time: %f seconds\n", end_time - start_time);

    // Print updated distances
    printf("\nDistances after updates (delete 0-1, insert 2-3):\n");
    for (idx_t i = 0; i < 10 && i < num_vertices; i++) {
        double d = (dist[i] == INF) ? -1.0 : dist[i];
        printf("Vertex %d: Distance = %.6f, Parent = %d\n", i, d, parent[i]);
    }

    // Cleanup
    free(dist);
    free(parent);
    free(affected);
    free(affected_del);
    free_graph(&graph);
    free(part);

    return 0;
}

// Read edge list and build CSR
void read_edge_list(const char *filename, idx_t *num_vertices, idx_t *num_edges, idx_t **xadj, idx_t **adjncy, double **weights) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        exit(1);
    }

    // First pass: count vertices and edges
    idx_t max_vertex = 0, edge_count = 0;
    idx_t u, v;
    double w;
    while (fscanf(fp, "%d %d %lf", &u, &v, &w) == 3) {
        if (u > max_vertex) max_vertex = u;
        if (v > max_vertex) max_vertex = v;
        edge_count++;
    }
    *num_vertices = max_vertex + 1;
    *num_edges = edge_count;

    // Temporary arrays
    idx_t *src = malloc(*num_edges * sizeof(idx_t));
    idx_t *dst = malloc(*num_edges * sizeof(idx_t));
    double *wgt = malloc(*num_edges * sizeof(double));

    // Second pass: store edges
    rewind(fp);
    for (idx_t i = 0; i < *num_edges; i++) {
        fscanf(fp, "%d %d %lf", &src[i], &dst[i], &wgt[i]);
    }
    fclose(fp);

    // Build CSR (undirected graph)
    idx_t *degree = calloc(*num_vertices, sizeof(idx_t));
    for (idx_t i = 0; i < *num_edges; i++) {
        degree[src[i]]++;
        if (src[i] != dst[i]) degree[dst[i]]++;
    }

    *xadj = malloc((*num_vertices + 1) * sizeof(idx_t));
    (*xadj)[0] = 0;
    for (idx_t i = 0; i < *num_vertices; i++) {
        (*xadj)[i + 1] = (*xadj)[i] + degree[i];
    }

    idx_t total_edges = (*xadj)[*num_vertices];
    *adjncy = malloc(total_edges * sizeof(idx_t));
    *weights = malloc(total_edges * sizeof(double));

    memset(degree, 0, *num_vertices * sizeof(idx_t));
    for (idx_t i = 0; i < *num_edges; i++) {
        idx_t s = src[i], d = dst[i];
        double w = wgt[i];
        idx_t pos = (*xadj)[s] + degree[s];
        (*adjncy)[pos] = d;
        (*weights)[pos] = w;
        degree[s]++;
        if (s != d) {
            pos = (*xadj)[d] + degree[d];
            (*adjncy)[pos] = s;
            (*weights)[pos] = w;
            degree[d]++;
        }
    }

    free(src);
    free(dst);
    free(wgt);
    free(degree);
}

// Partition graph using METIS
void partition_graph(idx_t num_vertices, idx_t *xadj, idx_t *adjncy, int num_parts, idx_t *part) {
    idx_t ncon = 1, objval;
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;

    int ret = METIS_PartGraphRecursive(&num_vertices, &ncon, xadj, adjncy, NULL, NULL, NULL,
                                       &num_parts, NULL, NULL, options, &objval, part);
    if (ret != METIS_OK) {
        fprintf(stderr, "METIS partitioning failed: %d\n", ret);
        exit(1);
    }
}

// Priority Queue implementation
PriorityQueue* pq_create(idx_t capacity) {
    PriorityQueue *pq = malloc(sizeof(PriorityQueue));
    pq->nodes = malloc(capacity * sizeof(PQNode));
    pq->positions = malloc(capacity * sizeof(idx_t));
    for (idx_t i = 0; i < capacity; i++) {
        pq->positions[i] = -1;
    }
    pq->size = 0;
    pq->capacity = capacity;
    return pq;
}

void pq_insert(PriorityQueue *pq, idx_t vertex, double dist) {
    if (pq->positions[vertex] != -1) {
        idx_t pos = pq->positions[vertex];
        if (dist < pq->nodes[pos].dist) {
            pq->nodes[pos].dist = dist;
            while (pos > 0 && pq->nodes[pos].dist < pq->nodes[(pos - 1) / 2].dist) {
                PQNode temp = pq->nodes[pos];
                pq->nodes[pos] = pq->nodes[(pos - 1) / 2];
                pq->nodes[(pos - 1) / 2] = temp;
                pq->positions[pq->nodes[pos].vertex] = pos;
                pq->positions[pq->nodes[(pos - 1) / 2].vertex] = (pos - 1) / 2;
                pos = (pos - 1) / 2;
            }
        }
    } else {
        idx_t pos = pq->size;
        pq->nodes[pos].vertex = vertex;
        pq->nodes[pos].dist = dist;
        pq->positions[vertex] = pos;
        pq->size++;
        while (pos > 0 && pq->nodes[pos].dist < pq->nodes[(pos - 1) / 2].dist) {
            PQNode temp = pq->nodes[pos];
            pq->nodes[pos] = pq->nodes[(pos - 1) / 2];
            pq->nodes[(pos - 1) / 2] = temp;
            pq->positions[pq->nodes[pos].vertex] = pos;
            pq->positions[pq->nodes[(pos - 1) / 2].vertex] = (pos - 1) / 2;
            pos = (pos - 1) / 2;
        }
    }
}

idx_t pq_extract_min(PriorityQueue *pq, double *dist) {
    if (pq->size == 0) return -1;
    idx_t min_vertex = pq->nodes[0].vertex;
    *dist = pq->nodes[0].dist;
    pq->positions[min_vertex] = -1;
    pq->size--;
    if (pq->size > 0) {
        pq->nodes[0] = pq->nodes[pq->size];
        pq->positions[pq->nodes[0].vertex] = 0;
        idx_t pos = 0;
        while (2 * pos + 1 < pq->size) {
            idx_t child = 2 * pos + 1;
            if (child + 1 < pq->size && pq->nodes[child + 1].dist < pq->nodes[child].dist) {
                child++;
            }
            if (pq->nodes[pos].dist > pq->nodes[child].dist) {
                PQNode temp = pq->nodes[pos];
                pq->nodes[pos] = pq->nodes[child];
                pq->nodes[child] = temp;
                pq->positions[pq->nodes[pos].vertex] = pos;
                pq->positions[pq->nodes[child].vertex] = child;
                pos = child;
            } else {
                break;
            }
        }
    }
    return min_vertex;
}

// Compute initial SSSP using Dijkstra's algorithm
void compute_sssp(Graph *graph, idx_t source, double *dist, idx_t *parent, int num_threads) {
    PriorityQueue *pq = pq_create(graph->num_vertices);
    for (idx_t i = 0; i < graph->num_vertices; i++) {
        dist[i] = INF;
        parent[i] = -1;
    }
    dist[source] = 0.0;
    pq_insert(pq, source, 0.0);

    while (pq->size > 0) {
        double current_dist;
        idx_t u = pq_extract_min(pq, &current_dist);
        if (current_dist > dist[u]) continue;

        for (idx_t j = graph->xadj[u]; j < graph->xadj[u + 1]; j++) {
            idx_t v = graph->adjncy[j];
            double w = graph->weights[j];
            double new_dist = dist[u] + w;
            if (new_dist < dist[v]) {
                dist[v] = new_dist;
                parent[v] = u;
                pq_insert(pq, v, new_dist);
            }
        }
    }
    free_pq(pq);
}

// Step 1: Identify affected vertices
void identify_affected(Graph *graph, EdgeChange *changes, int num_changes, double *dist, idx_t *parent,
                      bool *affected, bool *affected_del, int num_threads) {
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for (int i = 0; i < num_changes; i++) {
        idx_t u = changes[i].u;
        idx_t v = changes[i].v;
        bool is_del = changes[i].is_deleted;

        if (is_del) {
            if (parent[v] == u) {
                dist[v] = INF;
                parent[v] = -1;
                affected_del[v] = true;
                affected[v] = true;
            } else if (parent[u] == v) {
                dist[u] = INF;
                parent[u] = -1;
                affected_del[u] = true;
                affected[u] = true;
            }
        } else {
            double w = changes[i].weight;
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                parent[v] = u;
                affected[v] = true;
            } else if (dist[v] + w < dist[u]) {
                dist[u] = dist[v] + w;
                parent[u] = v;
                affected[u] = true;
            }
        }
    }
}

// Step 2: Update affected subgraph
void update_subgraph(Graph *graph, double *dist, idx_t *parent, bool *affected, bool *affected_del,
                     int num_threads) {
    bool change = true;
    omp_set_num_threads(num_threads);

    // Propagate deletions
    while (change) {
        change = false;
        #pragma omp parallel for schedule(dynamic)
        for (idx_t v = 0; v < graph->num_vertices; v++) {
            if (affected_del[v]) {
                for (idx_t j = graph->xadj[v]; j < graph->xadj[v + 1]; j++) {
                    idx_t child = graph->adjncy[j];
                    if (parent[child] == v) {
                        dist[child] = INF;
                        parent[child] = -1;
                        affected_del[child] = true;
                        affected[child] = true;
                        change = true;
                    }
                }
                affected_del[v] = false;
            }
        }
    }

    // Update distances
    change = true;
    while (change) {
        change = false;
        #pragma omp parallel for schedule(dynamic)
        for (idx_t v = 0; v < graph->num_vertices; v++) {
            if (affected[v]) {
                affected[v] = false;
                for (idx_t j = graph->xadj[v]; j < graph->xadj[v + 1]; j++) {
                    idx_t n = graph->adjncy[j];
                    double w = graph->weights[j];
                    if (dist[n] + w < dist[v]) {
                        dist[v] = dist[n] + w;
                        parent[v] = n;
                        affected[v] = true;
                        change = true;
                    }
                    if (dist[v] + w < dist[n]) {
                        dist[n] = dist[v] + w;
                        parent[n] = v;
                        affected[n] = true;
                        change = true;
                    }
                }
            }
        }
    }
}

// Cleanup functions
void free_graph(Graph *graph) {
    free(graph->xadj);
    free(graph->adjncy);
    free(graph->weights);
}

void free_pq(PriorityQueue *pq) {
    free(pq->nodes);
    free(pq->positions);
    free(pq);
}
