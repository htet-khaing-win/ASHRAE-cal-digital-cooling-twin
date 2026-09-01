# Deployment — M8 placeholder

Nothing here yet. M8's acceptance criteria (`docs/MILESTONES.md`):

- [ ] `docker compose up` brings gateway + twin_mcp + Redis up
- [ ] kind cluster runs gateway (**2 replicas**) and twin_mcp
- [ ] `policies/policy.yaml` mounted as a ConfigMap, hot-reloadable
- [ ] Receipt HMAC key mounted as a Secret — never in an image or config
- [ ] Non-root, `readOnlyRootFilesystem`, all capabilities dropped

**Redis is required from M2, not M8** (ADR-0005). Two replicas with an
in-memory nonce store is not a performance problem, it is a T-09 replay
bypass: a nonce consumed on pod A is unknown to pod B.

```
deploy/
  docker-compose.yml    gateway · twin_mcp · redis
  k8s/                  Deployment, Service, ConfigMap, Secret, NetworkPolicy
```
