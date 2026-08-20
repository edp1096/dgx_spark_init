package main

import (
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
	"mediaapp/internal/server"
	webassets "mediaapp/web"
)

func main() {
	configPath := "media.yaml"
	cfg, created, err := config.Load(configPath)
	if err != nil {
		log.Fatal(err)
	}
	if created {
		log.Printf("created sample configuration: %s", configPath)
	}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		log.Fatal(err)
	}
	mediaServer := server.New(cfg, store, webassets.Files(), configPath)
	resumed, failed := mediaServer.ResumeInterruptedJobs()
	if resumed > 0 || failed > 0 {
		log.Printf("restart recovery: resumed %d subtitle jobs, marked %d other jobs failed", resumed, failed)
	}
	srv := &http.Server{Addr: cfg.Listen, Handler: mediaServer.Handler()}
	go func() {
		result, cleanupErr := mediaServer.CleanupStaleMediaTemp()
		if cleanupErr != nil {
			log.Printf("temporary media cleanup skipped: %v", cleanupErr)
		} else if result.RemovedDirectories > 0 {
			log.Printf("removed %d stale temporary media directories (%d bytes)", result.RemovedDirectories, result.RemovedBytes)
		}
	}()
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	go func() { <-stop; _ = srv.Close() }()
	log.Printf("media app listening on http://%s", cfg.Listen)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}
