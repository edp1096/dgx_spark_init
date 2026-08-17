package main

import (
	"fmt"
	"os"

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

	client := llm.New(cfg.Model.Endpoint, cfg.Model.DefaultModel, cfg.Model.APIKey)
	srv, err := server.New(cfg, config.DefaultPath, store, client, chat.WebDist)
	if err != nil {
		fmt.Fprintf(os.Stderr, "server: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("SparkTalk: http://%s (model endpoint: %s)\n", cfg.Server.ListenAddr, cfg.Model.Endpoint)
	if err := srv.ListenAndServe(); err != nil {
		fmt.Fprintf(os.Stderr, "server: %v\n", err)
		os.Exit(1)
	}
}
