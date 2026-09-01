package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
	"mediaapp/internal/server"
	webassets "mediaapp/web"
)

func main() {
	configPath := defaultConfigPath()
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
	if cancelled := mediaServer.CancelActiveMediaPreparations(); cancelled > 0 {
		log.Printf("restart recovery: stopped %d stale media preparation requests", cancelled)
	}
	resumed, failed := mediaServer.ResumeInterruptedJobs()
	if resumed > 0 || failed > 0 {
		log.Printf("restart recovery: resumed %d durable queued jobs, marked %d interrupted jobs failed", resumed, failed)
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
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()
		if err := mediaServer.PreparePortraitLab(ctx); err != nil {
			log.Printf("PORTRAIT LAB preparation deferred: %v", err)
		} else {
			log.Printf("PORTRAIT LAB ready: v1.7.1 original HTML")
		}
	}()
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()
		if err := mediaServer.PreparePromptWildcards(ctx); err != nil {
			log.Printf("prompt wildcard preparation deferred: %v", err)
		} else {
			log.Printf("prompt wildcards ready: Crocody/mymuse muse + muse(no_camera) + Style")
		}
	}()
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-stop
		if cancelled := mediaServer.CancelActiveMediaPreparations(); cancelled > 0 {
			log.Printf("shutdown: stopped %d active media preparation requests", cancelled)
		}
		_ = srv.Close()
	}()
	log.Printf("media app listening on http://%s", cfg.Listen)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}

// defaultConfigPath keeps configuration and durable data beside the executable
// rather than tying them to the caller's working directory. A binary in dist/
// therefore always uses dist/media.yaml and dist/data.
func defaultConfigPath() string {
	workingDirectory, err := os.Getwd()
	if err != nil {
		workingDirectory = "."
	}
	executable, err := os.Executable()
	if err == nil {
		if resolved, resolveErr := filepath.EvalSymlinks(executable); resolveErr == nil {
			executable = resolved
		}
	}
	return resolveDefaultConfigPath(executable, workingDirectory, os.Getenv("SPARKMEDIA_CONFIG_PATH"))
}

func resolveDefaultConfigPath(executable, workingDirectory, explicit string) string {
	if path := strings.TrimSpace(explicit); path != "" {
		if absolute, err := filepath.Abs(path); err == nil {
			return absolute
		}
		return path
	}
	if executable != "" {
		return filepath.Join(filepath.Dir(executable), "media.yaml")
	}
	if absolute, err := filepath.Abs(filepath.Join(workingDirectory, "media.yaml")); err == nil {
		return absolute
	}
	return filepath.Join(workingDirectory, "media.yaml")
}
