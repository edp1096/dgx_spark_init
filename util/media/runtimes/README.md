# SparkMedia runtimes

This directory is owned by SparkMedia. Each allowlisted runtime contains its
own Compose definition and build context; runtime control never reads from the
repository-level `compose_yaml` directory.

The backend accepts only the fixed runtime names defined in
`internal/server/runtime_manager.go`. It does not accept compose paths, service
names, or shell commands from HTTP clients.
