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
		if err := checkHealth(cfg); err != nil {
			log.Fatal(err)
		}
		return
	}
	api := newCollectorAPI(cfg)
	server := &http.Server{Addr: cfg.ListenAddr, Handler: api.routes(), ReadHeaderTimeout: 10 * time.Second, IdleTimeout: 60 * time.Second}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = server.Shutdown(shutdown)
	}()
	log.Printf("SparkTalk Extra Collector listening on %s", cfg.ListenAddr)
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}

func checkHealth(cfg config) error {
	_, port, err := net.SplitHostPort(cfg.ListenAddr)
	if err != nil {
		return err
	}
	client := &http.Client{Timeout: 3 * time.Second}
	response, err := client.Get("http://127.0.0.1:" + port + "/health")
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("health endpoint returned %s", response.Status)
	}
	return nil
}
