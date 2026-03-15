# Format Converter (Library)

## Overview

Internal C++ utilities for converting between graph formats. Used by the `graph_converter` CLI and other tools.

## Source

`core/util/format_converter.cuh`

## Functionality

- Edgelist ↔ ImmutableCSR
- ImmutableCSR ↔ Grid CSR tiled matrix
- Edgelist ↔ Bit-tiled matrix

## Usage

Include the header and call the appropriate conversion functions from `sics::matrixgraph::core::util::format_converter`. For CLI usage, see [GraphConverter.md](GraphConverter.md).
