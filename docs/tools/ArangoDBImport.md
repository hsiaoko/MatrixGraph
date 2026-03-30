# ArangoDB Import Guide (MatrixGraph)

This document explains how to import MatrixGraph exported JSON/JSONL data into
ArangoDB, with collection naming aligned to GAR runtime lookup:

`datasource_{graphId}_time_pivot_graph_{businessId}`.

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
