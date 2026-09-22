package main

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	chat "sparktalk"
	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/server"
)

func main() {
	cfg, generated, err := config.Load(config.DefaultPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "config: %v\n", err)
		os.Exit(1)
	}
	if generated {
		fmt.Printf("Created default config: %s\n", config.DefaultPath)
	}

	store, err := db.Open(cfg.Server.Database)
	if err != nil {
		fmt.Fprintf(os.Stderr, "database: %v\n", err)
		os.Exit(1)
	}
	defer store.Close()

	client := llm.New(cfg.Model.Endpoint, cfg.Model.DefaultModel, cfg.Model.APIKey, cfg.Model.ModelType).WithThinkingBudget(cfg.Model.ThinkingBudget)
	srv, err := server.New(cfg, config.DefaultPath, store, client, chat.WebDist)
	if err != nil {
		fmt.Fprintf(os.Stderr, "server: %v\n", err)
		os.Exit(1)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	shutdownDone := make(chan struct{})
	go func() {
		defer close(shutdownDone)
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		if err := srv.Shutdown(shutdown); err != nil {
			fmt.Fprintf(os.Stderr, "shutdown: %v\n", err)
		}
	}()
	fmt.Printf("SparkTalk: http://%s (model endpoint: %s)\n", cfg.Server.ListenAddr, cfg.Model.Endpoint)
	err = srv.ListenAndServe()
	if errors.Is(err, http.ErrServerClosed) {
		<-shutdownDone
	}
	if err != nil && !errors.Is(err, http.ErrServerClosed) {
		fmt.Fprintf(os.Stderr, "server: %v\n", err)
		os.Exit(1)
	}
}
