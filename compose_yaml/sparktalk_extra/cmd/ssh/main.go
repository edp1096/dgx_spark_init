package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	healthcheck := flag.Bool("healthcheck", false, "check the local HTTP health endpoint")
	flag.Parse()
	cfg, err := loadConfig()
	if err != nil {
		log.Fatal(err)
	}
	if *healthcheck {
		if err := checkHealth(cfg.ListenAddr); err != nil {
			log.Fatal(err)
		}
		return
	}
	service := newAPI(cfg)
	server := &http.Server{Addr: cfg.ListenAddr, Handler: service.routes(), ReadHeaderTimeout: 10 * time.Second, IdleTimeout: 60 * time.Second}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = server.Shutdown(shutdownCtx)
	}()
	log.Printf("SparkTalk Extra SSH listening on %s", cfg.ListenAddr)
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}

func checkHealth(listenAddr string) error {
	_, port, err := net.SplitHostPort(listenAddr)
	if err != nil {
		return fmt.Errorf("invalid listen address: %w", err)
	}
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get("http://127.0.0.1:" + port + "/health")
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health endpoint returned %s", resp.Status)
	}
	return nil
}
