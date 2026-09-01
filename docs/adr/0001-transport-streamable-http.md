# ADR-0001 — Streamable HTTP transport, with a REST shim deferred

**Status:** accepted · **Date:** 2026-09-01 · **Milestone:** M3

## Context
MCP offers stdio and streamable HTTP. The gateway must sit between two
processes, run as two Kubernetes replicas (M8), and be reachable by an
agent that is not a child process of it.

## Decision
Streamable HTTP for both hops: agent → gateway, gateway → twin_mcp.

## Alternatives
**stdio.** Simpler and the default for local MCP servers. Rejected: it
requires the client to spawn the server as a subprocess, which makes a
network gateway impossible and a second replica meaningless. It stays
useful for local Inspector work against `twin_mcp` alone.

**A REST API with an MCP shim.** Would make the gateway callable by
non-MCP clients and trivially curl-testable. Rejected for the MVP: it
doubles the surface to authorize, and two entry points to `apply` is
exactly the shape of bug ADR-0004 exists to prevent. If a REST facade is
ever added it must call the same decision path, not a parallel one.

## Consequences
- Needs a real HTTP server and its own timeout/retry handling
- Inspector works against both hops
- Two replicas become possible, which forces the state decision in ADR-0005
