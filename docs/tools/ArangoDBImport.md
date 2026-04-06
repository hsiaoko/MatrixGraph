# ArangoDB Import Guide (MatrixGraph / GAR Platform)

This document explains how to set up ArangoDB collections and import data
for the GAR platform (Go `gdbc` library + MatrixGraph runtime).

> **Key insight**: The `gdbc` library (`gitlab.grandhoo.com/module/gdbc`) uses
> a fixed collection naming convention based on `datasource_{graphId}`.
> All collections live in a single ArangoDB database chosen at runtime
> (e.g. `congress`, `stanford`, `_system`).

---

## Table of Contents

1. [Collection Naming Convention](#1-collection-naming-convention)
2. [Prerequisites](#2-prerequisites)
3. [Set Variables](#3-set-variables)
4. [Create Database](#4-create-database)
5. [Create Collections](#5-create-collections)
6. [Import Data](#6-import-data)
   - [6.1 Import `datasource_{graphId}_meta`](#61-import-datasourcegraphid_meta)
   - [6.2 Import `datasource_{graphId}_time_pivot_graph_{businessId}`](#62-import-datasourcegraphid_time_pivot_graph_businessid)
   - [6.3 Import label-specific vertex/edge collections (optional)](#63-import-label-specific-vertexedge-collections-optional)
7. [Verify Import](#7-verify-import)
8. [Export Edgelist CSV → ArangoDB JSON (graph-convert)](#8-export-edgelist-csv--arangodb-json-graph-convert)
9. [Running the Discover Demo](#9-running-the-discover-demo)
10. [Common Errors](#10-common-errors)
11. [Complete End-to-End Script](#11-complete-end-to-end-script)

---

## 1. Collection Naming Convention

The `gdbc` library expects the following collections for a given `graphId`:

| Collection Name Pattern | Example (`graphId=1`, `businessId=1`) | Purpose |
|---|---|---|
| `datasource_{graphId}_meta` | `datasource_1_meta` | **Core.** Graph structure, business metadata, time split strategy, label counts |
| `datasource_{graphId}_time_pivot_graph_{businessId}` | `datasource_1_time_pivot_graph_1` | **Core.** Pivot graph documents (one doc per pivot) |
| `datasource_{graphId}_graph` | `datasource_1_graph` | ArangoDB graph definition (for AQL traversals) |
| `datasource_{graphId}_v{N}_{label}` | `datasource_1_v0_chemical` | Vertex data by label (used by `ReadVerticesInStream`) |
| `datasource_{graphId}_e{N}_{label}` | `datasource_1_e0_relationship` | Edge data by label (used by `ReadEdgesInStream`) |
| `datasource_{graphId}_v{N}_{label}_history` | `datasource_1_v0_chemical_history` | Vertex history (temporal attributes) |
| `datasource_{graphId}_e{N}_{label}_history` | `datasource_1_e0_relationship_history` | Edge history (temporal attributes) |
| `datasource_{graphId}_view` | `datasource_1_view` | ArangoDB search view (optional) |

> **Minimum to run the discover demo**: `datasource_{graphId}_meta` +
> `datasource_{graphId}_time_pivot_graph_{businessId}`.

### Format Constants (from `gdbc` source)

```go
metaNameFormat                 = "datasource_%s_meta"
graphNameFormat                = "datasource_%s_graph"
viewNameFormat                 = "datasource_%s_view"
collectionVertexFormat         = "datasource_%s_v%d_%s"
collectionEdgeFormat           = "datasource_%s_e%d_%s"
collectionVertexHistoryFormat  = "datasource_%s_v%d_%s_history"
collectionEdgeHistoryFormat    = "datasource_%s_e%d_%s_history"
collectionTimePivotGraphFormat = "datasource_%s_time_pivot_graph_%s"
```

---

## 2. Prerequisites

- `arangosh` CLI is available.
- `arangoimport` CLI is available (optional, for bulk JSONL import).
- Valid ArangoDB credentials.
- Source data files (JSONL or JSON) if importing from files.

---

## 3. Set Variables

```bash
# ── Connection ──────────────────────────────────────────────────────────────
export ARANGO_ENDPOINT="tcp://192.168.51.10:8529"
export ARANGO_USER="root"
export ARANGO_PASSWORD="123456"
export ARANGO_DB="congress"

# ── Graph identifiers ───────────────────────────────────────────────────────
export GRAPH_ID="1"          # Used in collection names as datasource_{graphId}_*
export BUSINESS_ID="1"       # Used in time-pivot collection name

# ── Derived collection names ────────────────────────────────────────────────
export META_COL="datasource_${GRAPH_ID}_meta"
export PIVOT_COL="datasource_${GRAPH_ID}_time_pivot_graph_${BUSINESS_ID}"
export GRAPH_COL="datasource_${GRAPH_ID}_graph"

# ── Input data paths (adjust to your environment) ──────────────────────────
export GRAPH_STRUCTURE_JSON="/path/to/graph_structure.json"
export PIVOT_GRAPHS_JSONL="/path/to/pivot_graphs.jsonl"
```

---

## 4. Create Database

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
  print("created database: " + name);
} else {
  print("database already exists: " + name);
}
'
```

---

## 5. Create Collections

```bash
arangosh \
  --server.endpoint "${ARANGO_ENDPOINT}" \
  --server.username "${ARANGO_USER}" \
  --server.password "${ARANGO_PASSWORD}" \
  --server.database "${ARANGO_DB}" \
  --javascript.execute-string '
var cols = [
  "'"${META_COL}"'",
  "'"${PIVOT_COL}"'",
  "'"${GRAPH_COL}"'"
];
cols.forEach(function(name) {
  if (!db._collection(name)) {
    db._createDocumentCollection(name);
    print("created: " + name);
  } else {
    print("exists:  " + name);
  }
});
'
```

---

## 6. Import Data

### 6.1 Import `datasource_{graphId}_meta`

This is the **most critical** collection. It stores multiple documents, each
identified by `_key`:

| `_key` value | Purpose | Required by |
|---|---|---|
| `"graph"` | Graph structure (vertices, edges, attributes) | `GetGraphStructure()` |
| `{businessId}` | Business metadata + `TimeBaseSplitStrategy` | `ReadTimeBaseSplitStrategy()`, `ReadTimePivotMinMaxTime()` |
| `"label_count"` | Vertex/edge counts per label (optional) | Statistics display |
| `"label_collection"` | Label → collection name mapping (optional) | Advanced label queries |
| `"label_history"` | Label → history collection mapping (optional) | History queries |

#### 6.1.1 `"graph"` document

Stores the graph schema — vertex labels, edge labels, and their attributes.

**Structure** (must match `base.Graph` + `GraphStructure`):

```json
{
  "_key": "graph",
  "DatasourceId": "1",
  "Timezone": "UTC",
  "Vertices": [
    {
      "label": "chemical",
      "attrs": [
        {
          "key": "DosingGuideline",
          "role": 5,
          "dataType": 2,
          "isList": false,
          "isDerivative": false
        }
      ]
    },
    {
      "label": "disease",
      "attrs": [
        {
          "key": "severity",
          "role": 5,
          "dataType": 2,
          "isList": false,
          "isDerivative": false
        }
      ]
    }
  ],
  "Edges": [
    {
      "label": "relationship",
      "srcLabel": "chemical",
      "dstLabel": "disease",
      "attrs": [
        {
          "key": "evidence",
          "role": 5,
          "dataType": 2,
          "isList": false,
          "isDerivative": false
        }
      ]
    }
  ]
}
```

> **`dataType` enum** (from `igeenum`):
> `1`=Float, `2`=String, `3`=Int, `4`=Bool, `5`=TimeType,
> `6`/`7`=List variants. Most text attributes use `2` (String).
>
> **`role` enum**: `5` is the common data role.

#### 6.1.2 `{businessId}` document (BusinessMeta)

Stores the time-split strategy, pivot label, k-hop config, and timestamps.

**Structure** (must match `BusinessMeta` + `base.TimeBaseSplitStrategy`):

```json
{
  "_key": "1",
  "_type": "business",
  "_status": "GraphUpdated",
  "_time": "2024-01-01 00:00:00",
  "_pivot_min_time": "2024-01-01 00:00:00",
  "_pivot_max_time": "2024-12-31 23:59:59",
  "_strategy": {
    "businessId": "1",
    "pivotLabel": "chemical",
    "kHop": 2,
    "labelTimeKey": {
      "vertexMap": { "chemical": "_time", "disease": "_time" },
      "edgeMap": {}
    },
    "labelTimeSource": {
      "vertexMap": { "chemical": {}, "disease": {} },
      "edgeMap": {}
    },
    "labelTimeRange": {
      "vertexMap": {},
      "edgeMap": {}
    },
    "labelAttrsTrans": {
      "vertexMap": {},
      "edgeMap": {}
    },
    "pivotStartTimeStr": "2024-01-01 00:00:00"
  }
}
```

**Field explanations:**

| Field | Description |
|---|---|
| `_key` | Must equal `businessId` (e.g. `"1"`) |
| `_type` | Must be `"business"` |
| `_status` | `"GraphUpdated"` (ready) or `"Graph Updating"` |
| `_time` | Last import/update timestamp |
| `_strategy.pivotLabel` | The pivot vertex label (e.g. `"chemical"`) |
| `_strategy.kHop` | K-hop radius for pivot subgraph extraction |
| `_strategy.labelTimeKey.vertexMap` | Maps each vertex label to its time column name |
| `_strategy.pivotStartTimeStr` | Pivot start time for incremental updates |

#### 6.1.3 Import commands for meta documents

```bash
arangosh \
  --server.endpoint "${ARANGO_ENDPOINT}" \
  --server.username "${ARANGO_USER}" \
  --server.password "${ARANGO_PASSWORD}" \
  --server.database "${ARANGO_DB}" \
  --javascript.execute-string '
var col = db._collection("'"${META_COL}"'");

// ── 1. Graph structure ──────────────────────────────────────────────
col.save({
  _key: "graph",
  DatasourceId: "'"${GRAPH_ID}"'",
  Timezone: "UTC",
  Vertices: [
    {
      label: "chemical",
      attrs: [
        { key: "DosingGuideline", role: 5, dataType: 2, isList: false, isDerivative: false }
      ]
    },
    {
      label: "disease",
      attrs: [
        { key: "severity", role: 5, dataType: 2, isList: false, isDerivative: false }
      ]
    }
  ],
  Edges: [
    {
      label: "relationship",
      srcLabel: "chemical",
      dstLabel: "disease",
      attrs: [
        { key: "evidence", role: 5, dataType: 2, isList: false, isDerivative: false }
      ]
    }
  ]
});
print("Inserted graph structure (_key=graph)");

// ── 2. Business metadata ────────────────────────────────────────────
col.save({
  _key: "'"${BUSINESS_ID}"'",
  _type: "business",
  _status: "GraphUpdated",
  _time: "2024-01-01 00:00:00",
  _pivot_min_time: "2024-01-01 00:00:00",
  _pivot_max_time: "2024-12-31 23:59:59",
  _strategy: {
    businessId: "'"${BUSINESS_ID}"'",
    pivotLabel: "chemical",
    kHop: 2,
    labelTimeKey: {
      vertexMap: { "chemical": "_time", "disease": "_time" },
      edgeMap: {}
    },
    labelTimeSource: {
      vertexMap: { "chemical": {}, "disease": {} },
      edgeMap: {}
    },
    labelTimeRange: { vertexMap: {}, edgeMap: {} },
    labelAttrsTrans:  { vertexMap: {}, edgeMap: {} },
    pivotStartTimeStr: "2024-01-01 00:00:00"
  }
});
print("Inserted business meta (_key='"${BUSINESS_ID}"')");
'
```

> **If you have a `graph_structure.json` file** from `graph-convert`, import it with:
>
> ```bash
> arangosh \
>   --server.endpoint "${ARANGO_ENDPOINT}" \
>   --server.username "${ARANGO_USER}" \
>   --server.password "${ARANGO_PASSWORD}" \
>   --server.database "${ARANGO_DB}" \
>   --javascript.execute-string '
> var fs = require("fs");
> var data = JSON.parse(fs.readFileSync("'"${GRAPH_STRUCTURE_JSON}"'"));
> var col = db._collection("'"${META_COL}"'");
> data._key = "graph";
> // Ensure DatasourceId is set
> if (!data.DatasourceId) { data.DatasourceId = "'"${GRAPH_ID}"'"; }
> col.save(data);
> print("Imported graph_structure.json");
> '
> ```

---

### 6.2 Import `datasource_{graphId}_time_pivot_graph_{businessId}`

Each document represents one pivot subgraph. The `_key` is the pivot graph ID.

**Document structure** (must match `base.PivotGraphDoc`):

```json
{
  "_key": "pg_0",
  "_time": "2024-01-01 00:00:00",
  "_pivot_time": "2024-01-01 00:00:00",
  "graph": {
    "pivotGraphId": "pg_0",
    "pivotLabel": "chemical",
    "vertices": [
      {
        "label": "chemical",
        "id": "chem_001",
        "time": "2024-01-01 00:00:00",
        "attrs": [
          { "key": "DosingGuideline", "value": "Standard" }
        ]
      },
      {
        "label": "disease",
        "id": "disease_001",
        "time": "2024-01-01 00:00:00",
        "attrs": [
          { "key": "severity", "value": "moderate" }
        ]
      }
    ],
    "edges": [
      {
        "label": "relationship",
        "srcLabel": "chemical",
        "dstLabel": "disease",
        "id": "rel_001",
        "srcId": "chem_001",
        "dstId": "disease_001",
        "time": "2024-01-01 00:00:00",
        "attrs": [
          { "key": "evidence", "value": "clinical_trial" }
        ]
      }
    ]
  }
}
```

#### 6.2.1 Import from JSONL file (using `arangoimport`)

If you have a `pivot_graphs.jsonl` file (one JSON object per line):

```bash
arangoimport \
  --server.endpoint "${ARANGO_ENDPOINT}" \
  --server.database "${ARANGO_DB}" \
  --server.username "${ARANGO_USER}" \
  --server.password "${ARANGO_PASSWORD}" \
  --file "${PIVOT_GRAPHS_JSONL}" \
  --type jsonl \
  --collection "${PIVOT_COL}" \
  --overwrite true
```

#### 6.2.2 Import sample data via arangosh

```bash
arangosh \
  --server.endpoint "${ARANGO_ENDPOINT}" \
  --server.username "${ARANGO_USER}" \
  --server.password "${ARANGO_PASSWORD}" \
  --server.database "${ARANGO_DB}" \
  --javascript.execute-string '
var col = db._collection("'"${PIVOT_COL}"'");

col.save({
  _key: "pg_0",
  _time: "2024-01-01 00:00:00",
  _pivot_time: "2024-01-01 00:00:00",
  graph: {
    pivotGraphId: "pg_0",
    pivotLabel: "chemical",
    vertices: [
      {
        label: "chemical",
        id: "chem_001",
        time: "2024-01-01 00:00:00",
        attrs: [
          { key: "DosingGuideline", value: "Standard" }
        ]
      },
      {
        label: "disease",
        id: "disease_001",
        time: "2024-01-01 00:00:00",
        attrs: [
          { key: "severity", value: "moderate" }
        ]
      }
    ],
    edges: [
      {
        label: "relationship",
        srcLabel: "chemical",
        dstLabel: "disease",
        id: "rel_001",
        srcId: "chem_001",
        dstId: "disease_001",
        time: "2024-01-01 00:00:00",
        attrs: [
          { key: "evidence", value: "clinical_trial" }
        ]
      }
    ]
  }
});
print("Inserted pivot graph pg_0");
print("Total pivots: " + col.count());
'
```

---

### 6.3 Import label-specific vertex/edge collections (optional)

These collections (`datasource_{graphId}_v{N}_{label}`, etc.) are used by
`ReadVerticesInStream` / `ReadEdgesInStream` for streaming reads. They are
**not required** for the discover demo which uses `ReadTimePivotGraphByIdLimitLabels`.

**Vertex document format** (`datasource_{graphId}_v{N}_{label}`):

```json
{
  "_key": "chem_001",
  "label": "chemical",
  "DosingGuideline_": "Standard",
  "_time": "2024-01-01 00:00:00",
  "_source": "import"
}
```

**Edge document format** (`datasource_{graphId}_e{N}_{label}`):

```json
{
  "_key": "rel_001",
  "_from": "datasource_1_v0_chemical/chem_001",
  "_to": "datasource_1_v1_disease/disease_001",
  "label": "relationship",
  "evidence_": "clinical_trial",
  "_time": "2024-01-01 00:00:00",
  "_source": "import"
}
```

> **Note**: Attribute keys are suffixed with `_` (e.g. `DosingGuideline_`)
> in these collections. The `gdbc` library strips the suffix when reading.

---

## 7. Verify Import

```bash
arangosh \
  --server.endpoint "${ARANGO_ENDPOINT}" \
  --server.username "${ARANGO_USER}" \
  --server.password "${ARANGO_PASSWORD}" \
  --server.database "${ARANGO_DB}" \
  --javascript.execute-string '
// Check meta collection
var meta = db._collection("'"${META_COL}"'");
print("=== " + "'"${META_COL}"'" + " ===");
print("document count: " + meta.count());
meta.all().toArray().forEach(function(d) {
  print("  _key=" + d._key + (d._type ? " (_type=" + d._type + ")" : ""));
});

// Check pivot collection
var piv = db._collection("'"${PIVOT_COL}"'");
print("=== " + "'"${PIVOT_COL}"'" + " ===");
print("document count: " + piv.count());

// Sample one pivot document
var sample = piv.firstExample({});
if (sample) {
  print("sample pivot _key: " + sample._key);
  if (sample.graph) {
    print("  vertices: " + sample.graph.vertices.length);
    print("  edges: " + sample.graph.edges.length);
  }
}
'
```

---

## 8. Export Edgelist CSV → ArangoDB JSON (graph-convert)

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

| Mode | Description |
|---|---|
| `single` (default) | All vertices/edges in one pivot graph |
| `source` | One pivot graph per source vertex (outgoing edges) |
| `k_hop` | One pivot graph per vertex (k-hop neighborhood) |

### K-Hop Subgraph Mode

```bash
# 2-hop neighborhoods
graph-convert \
  --convert_mode=edgelistcsv2arangodbjson \
  -i graph.csv \
  -o arangodb_output \
  --pivot_mode=k_hop \
  --k_hop=2 \
  --graph_id=1 \
  --business_id=1
```

**Output files:**
- `graph_structure.json` — Global graph metadata → import as `_key="graph"` in meta collection
- `pivot_graph_ids.jsonl` — One line per pivot graph ID
- `pivot_graphs.jsonl` — One line per pivot, contains `vertices` + `edges`

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

---

## 9. Running the Discover Demo

After collections are set up, run the demo:

```bash
# Set environment variables to match your ArangoDB setup
export ARANGODB_ENDPOINTS="http://192.168.51.10:8529/"
export ARANGODB_USER="root"
export ARANGODB_PASSWORD="123456"
export ARANGODB_DATABASE="congress"
export ARANGODB_GRAPH_ID="1"
export ARANGODB_BUS_ID="1"

# Run discover-pattern demo
CGO_ENABLED=1 go run -tags=matrixgraph ./cmd/matrixgraph_demo -demo=discover
```

> **Important**: `ARANGODB_GRAPH_ID` and `ARANGODB_BUS_ID` must match the
> `graphId`/`businessId` used when creating the collections.
> e.g. `GRAPH_ID=1, BUSINESS_ID=1` → collections `datasource_1_meta`
> and `datasource_1_time_pivot_graph_1`.

---

## 10. Common Errors

| Error | Cause | Fix |
|---|---|---|
| `ArangoError 1228: database not found` | Database does not exist | Create via `_system` first (see [§4](#4-create-database)) |
| `ArangoError 1208: illegal name: database name invalid` | Invalid database name | Use a legal name (lowercase, no special chars) |
| `ArangoError 1203: collection or view not found` | Collection missing | Create collections (see [§5](#5-create-collections)) |
| `datasource_X_meta not found` | Meta collection missing | Most common error — must exist with `_key="graph"` and `_key="{businessId}"` documents |
| `readBusinessMeta failed` | Business meta document missing | Insert document with `_key="{businessId}"` containing `_strategy` |
| `panic: nil pointer dereference` in `ReadTimeBaseSplitStrategy` | `_strategy` field is nil or missing in business meta | Ensure `_strategy` object has all required sub-fields (see [§6.1.2](#612-businessid-document-businessmeta)) |
| `not connected` | Connecting to non-existent database | Verify database name and credentials |
| `get attr key failed label=xxx` | Vertex label not in `graph_structure.json` `Vertices` | Add the label to `Vertices` in the `_key="graph"` document |
| `collection or view not found, 1203, true` | Mismatch between runtime `graphId`/`businessId` and actual collection names | Verify `GRAPH_ID`/`BUSINESS_ID` env vars match collection naming |

---

## 11. Complete End-to-End Script

Copy and customize this script to set up everything from scratch:

```bash
#!/bin/bash
# setup_arangodb_for_gar.sh
# Run once to create database, collections, and sample data.

set -e

# ── Configuration ─────────────────────────────────────────────────────
ARANGO_ENDPOINT="tcp://192.168.51.10:8529"
ARANGO_USER="root"
ARANGO_PASSWORD="123456"
ARANGO_DB="congress"
GRAPH_ID="1"
BUSINESS_ID="1"

META_COL="datasource_${GRAPH_ID}_meta"
PIVOT_COL="datasource_${GRAPH_ID}_time_pivot_graph_${BUSINESS_ID}"

# ── Step 1: Create database ──────────────────────────────────────────
echo ">>> Creating database '${ARANGO_DB}' ..."
arangosh \
  --server.endpoint "${ARANGO_ENDPOINT}" \
  --server.username "${ARANGO_USER}" \
  --server.password "${ARANGO_PASSWORD}" \
  --server.database _system \
  --javascript.execute-string "
var name = '${ARANGO_DB}';
var dbs = db._databases();
if (dbs.indexOf(name) === -1) {
  db._createDatabase(name);
  print('created database: ' + name);
} else {
  print('database already exists: ' + name);
}
"

# ── Step 2: Create collections ───────────────────────────────────────
echo ">>> Creating collections ..."
arangosh \
  --server.endpoint "${ARANGO_ENDPOINT}" \
  --server.username "${ARANGO_USER}" \
  --server.password "${ARANGO_PASSWORD}" \
  --server.database "${ARANGO_DB}" \
  --javascript.execute-string "
var cols = ['${META_COL}', '${PIVOT_COL}', 'datasource_${GRAPH_ID}_graph'];
cols.forEach(function(name) {
  if (!db._collection(name)) {
    db._createDocumentCollection(name);
    print('created: ' + name);
  } else {
    print('exists:  ' + name);
  }
});
"

# ── Step 3: Import meta documents ────────────────────────────────────
echo ">>> Importing meta documents ..."
arangosh \
  --server.endpoint "${ARANGO_ENDPOINT}" \
  --server.username "${ARANGO_USER}" \
  --server.password "${ARANGO_PASSWORD}" \
  --server.database "${ARANGO_DB}" \
  --javascript.execute-string "
var col = db._collection('${META_COL}');

// Graph structure
col.save({
  _key: 'graph',
  DatasourceId: '${GRAPH_ID}',
  Timezone: 'UTC',
  Vertices: [
    { label: 'chemical', attrs: [
        { key: 'DosingGuideline', role: 5, dataType: 2, isList: false, isDerivative: false }
      ]
    },
    { label: 'disease', attrs: [
        { key: 'severity', role: 5, dataType: 2, isList: false, isDerivative: false }
      ]
    }
  ],
  Edges: [
    { label: 'relationship', srcLabel: 'chemical', dstLabel: 'disease', attrs: [
        { key: 'evidence', role: 5, dataType: 2, isList: false, isDerivative: false }
      ]
    }
  ]
});
print('Inserted _key=graph');

// Business metadata
col.save({
  _key: '${BUSINESS_ID}',
  _type: 'business',
  _status: 'GraphUpdated',
  _time: '2024-01-01 00:00:00',
  _pivot_min_time: '2024-01-01 00:00:00',
  _pivot_max_time: '2024-12-31 23:59:59',
  _strategy: {
    businessId: '${BUSINESS_ID}',
    pivotLabel: 'chemical',
    kHop: 2,
    labelTimeKey: {
      vertexMap: { 'chemical': '_time', 'disease': '_time' },
      edgeMap: {}
    },
    labelTimeSource: {
      vertexMap: { 'chemical': {}, 'disease': {} },
      edgeMap: {}
    },
    labelTimeRange:  { vertexMap: {}, edgeMap: {} },
    labelAttrsTrans: { vertexMap: {}, edgeMap: {} },
    pivotStartTimeStr: '2024-01-01 00:00:00'
  }
});
print('Inserted _key=${BUSINESS_ID}');
"

# ── Step 4: Import sample pivot data ─────────────────────────────────
echo ">>> Importing sample pivot data ..."
arangosh \
  --server.endpoint "${ARANGO_ENDPOINT}" \
  --server.username "${ARANGO_USER}" \
  --server.password "${ARANGO_PASSWORD}" \
  --server.database "${ARANGO_DB}" \
  --javascript.execute-string "
var col = db._collection('${PIVOT_COL}');
col.save({
  _key: 'pg_0',
  _time: '2024-01-01 00:00:00',
  _pivot_time: '2024-01-01 00:00:00',
  graph: {
    pivotGraphId: 'pg_0',
    pivotLabel: 'chemical',
    vertices: [
      { label: 'chemical', id: 'chem_001', time: '2024-01-01 00:00:00',
        attrs: [{ key: 'DosingGuideline', value: 'Standard' }] },
      { label: 'disease', id: 'disease_001', time: '2024-01-01 00:00:00',
        attrs: [{ key: 'severity', value: 'moderate' }] }
    ],
    edges: [
      { label: 'relationship', srcLabel: 'chemical', dstLabel: 'disease',
        id: 'rel_001', srcId: 'chem_001', dstId: 'disease_001',
        time: '2024-01-01 00:00:00',
        attrs: [{ key: 'evidence', value: 'clinical_trial' }] }
    ]
  }
});
print('Inserted pivot pg_0, total: ' + col.count());
"

echo ">>> Done! Setup complete."
echo "    Run: CGO_ENABLED=1 go run -tags=matrixgraph ./cmd/matrixgraph_demo -demo=discover"
```
