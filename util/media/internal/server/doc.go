// Package server assembles Spark Media's HTTP API and coordinates media jobs.
//
// Files are grouped by responsibility rather than size:
//   - *_create_http.go and *_http.go validate HTTP input and create jobs.
//   - *_queue.go owns scheduling and job state transitions.
//   - *_executor.go and *_job_execution.go reconstruct and run queued work.
//   - *_options.go contains the request contract shared by creation and execution.
//   - remote_*.go contains outbound engine and asset transport.
//   - subtitle_*.go follows the preparation, recognition, translation, and rendering pipeline.
//
// Handlers should not implement engine transport, and engine executors should not
// depend on HTTP request objects. Persisted job parameters are the boundary between them.
package server
