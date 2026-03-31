# ArangoDB Import Guide (MatrixGraph)

This document explains how to import MatrixGraph exported JSON/JSONL data into
ArangoDB, with collection naming aligned to GAR runtime lookup:

`datasource_{graphId}_time_pivot_graph_{businessId}`.

## 0. Export Edgelist to ArangoDB JSON

MatrixGraph provides a tool to convert edgelist CSV to ArangoDB JSON format.

### Basic Usage

```bash
graph-convert \
  --convert_mode=edgelistcsv2arangodbjson \
  -i <input_edgelist.csv> \
  -o <output_directory> \
  [--sep=separator] \
  [--keep_original_vid] \
  [--graph_id=<id>] \
  [--business_id=<id>] \
  [--pivot_mode=<mode>] \
  [--k_hop=<N>]
```

### Pivot Modes

The `--pivot_mode` option controls how pivot graphs are generated:

- **`single`** (default): All vertices and edges are stored in a single pivot graph (one row in `pivot_graphs.jsonl`).

- **`source`**: One pivot graph per source vertex. Each pivot contains all outgoing edges from that source vertex.

- **`k_hop`**: One pivot graph per vertex. Each pivot contains the k-hop outgoing neighborhood subgraph centered at that vertex (the pivot vertex and all vertices/edges reachable within k hops).

### K-Hop Subgraph Mode

When using `--pivot_mode=k_hop`, each pivot graph represents a vertex's k-hop neighborhood:

```bash
# Export with 2-hop neighborhoods (default k=2)
graph-convert \
  --convert_mode=edgelistcsv2arangodbjson \
  -i graph.csv \
  -o arangodb_output \
  --pivot_mode=k_hop \
  --k_hop=2 \
  --graph_id=1 \
  --business_id=1

# Export with 3-hop neighborhoods
graph-convert \
  --convert_mode=edgelistcsv2arangodbjson \
  -i graph.csv \
  -o arangodb_output \
  --pivot_mode=k_hop \
  --k_hop=3
```

**Output Structure (k_hop mode):**
- `graph_structure.json`: Global graph metadata (vertex/edge labels, counts)
- `pivot_graph_ids.jsonl`: One line per pivot graph with IDs
- `pivot_graphs.jsonl`: One line per pivot vertex, containing:
  - `pivot_graph_id`: Unique identifier (e.g., `pg_123` for pivot vertex 123)
  - `vertices`: All vertices reachable within k hops from the pivot
  - `edges`: All edges within the k-hop subgraph

### Additional Options

```bash
--sep=<separator>           # CSV separator (default: comma)
--keep_original_vid         # Keep original vertex IDs (no compression)
--graph_id=<id>             # Graph ID (default: 1)
--business_id=<id>          # Business ID (default: 1)
--import_time=<ts>          # Import timestamp (_time field)
--pivot_time=<ts>           # Business timestamp (_pivot_time field)
--default_vertex_label=<l>  # Default vertex label (default: "vertex")
--default_edge_label=<l>    # Default edge label (default: "relationship")
--random_vertex_labels      # Randomly assign labels within label_range
--label_range=<N>           # Range for random labels (default: 1)
```

## 1. Prerequisites

- `arangosh` is available.
- `arangoimport` is available.
- ArangoDB endpoint/username/password are valid.
- Export file exists, for example `pivot_graphs.jsonl`.

## 2. Parameterized Variables (No Hardcoded Paths)

Set runtime variables first:

```bash
export ARANGO_ENDPOINT="tcp://<host>:<port>"      # e.g. tcp://192.168.51.10:8529
export ARANGO_USER="<username>"                   # e.g. root
export ARANGO_PASSWORD="<password>"               # e.g. 123456
export ARANGO_DB="<database>"                     # e.g. stanford

export GRAPH_ID="<graphId>"                       # e.g. 1998649691503333376
export BUSINESS_ID="<businessId>"                 # e.g. 1998657722987319296

export INPUT_FILE="<path/to/pivot_graphs.jsonl>"  # e.g. /ssd_data/zhuxk/arangodb_graphs/stanford/pivot_graphs.jsonl
export COLLECTION="datasource_${GRAPH_ID}_time_pivot_graph_${BUSINESS_ID}"
```

## 3. Create Database (if missing)

Database creation must connect to `_system` first.

```bash
arangosh \
  --server.endpoint "${ARANGO_ENDPOINT}" \
  --server.username "${ARANGO_USER}" \
  --server.password "${ARANGO_PASSWORD}" \
  --server.database _system \
  --javascript.execute-string '
const name = "'"${ARANGO_DB}"'";
const dbs = db._databases();
if (dbs.indexOf(name) === -1) {
  db._createDatabase(name);
  print("created database:", name);
} else {
  print("database already exists:", name);
}
'
```

## 4. Create Collection (if missing)

Collection name follows:
`datasource_{graphId}_time_pivot_graph_{businessId}`.

```bash
arangosh \
  --server.endpoint "${ARANGO_ENDPOINT}" \
  --server.username "${ARANGO_USER}" \
  --server.password "${ARANGO_PASSWORD}" \
  --server.database "${ARANGO_DB}" \
  --javascript.execute-string '
const col = "'"${COLLECTION}"'";
if (!db._collection(col)) {
  db._createDocumentCollection(col);
  print("created collection:", col);
} else {
  print("collection exists:", col);
}
'
```

## 5. Import JSONL Data

```bash
arangoimport \
  --server.endpoint "${ARANGO_ENDPOINT}" \
  --server.database "${ARANGO_DB}" \
  --server.username "${ARANGO_USER}" \
  --server.password "${ARANGO_PASSWORD}" \
  --file "${INPUT_FILE}" \
  --type jsonl \
  --collection "${COLLECTION}" \
  --overwrite true
```

## 6. Verify Import

```bash
arangosh \
  --server.endpoint "${ARANGO_ENDPOINT}" \
  --server.username "${ARANGO_USER}" \
  --server.password "${ARANGO_PASSWORD}" \
  --server.database "${ARANGO_DB}" \
  --javascript.execute-string '
const col = "'"${COLLECTION}"'";
print("count =", db._collection(col).count());
printjson(db._query("FOR d IN @@c LIMIT 1 RETURN d", {"@c": col}).toArray());
'
```

## 7. Common Errors

- `ArangoError 1228: database not found`
  - Database does not exist. Create it via `_system` first.
- `ArangoError 1208: illegal name: database name invalid`
  - Database name is invalid. Use a legal name (e.g. `stanford`).
- `ArangoError 1203: collection or view not found`
  - Collection missing or naming mismatch with runtime graph/business IDs.
- `not connected`
  - Usually caused by connecting to a non-existent database.

## 8. Keep Runtime Params Consistent

When running GAR/matrixgraph demo, make sure these match imported data:

- `ARANGODB_ENDPOINTS` matches `ARANGO_ENDPOINT`
- `ARANGODB_DATABASE` matches `ARANGO_DB`
- `ARANGODB_GRAPH_ID` matches `GRAPH_ID`
- `ARANGODB_BUS_ID` matches `BUSINESS_ID`
