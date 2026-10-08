package server

import (
	"context"
	"fmt"
	"math"
	"sparktalk/internal/asr"
)

func (s *Server) acquireWorkload(ctx context.Context, target string, peak ...float64) (func(...error) error, error) {
	cfg, _ := s.snapshot()
	if s.runtime == nil || cfg.Runtime.Mode != "managed" {
		return func(...error) error { return nil }, nil
	}
	bundle := cfg.Runtime.ActiveBundle
	if bundle == "" {
		bundle = cfg.Runtime.Bundle
	}
	if target == "flux2" {
		if _, ok := videoRuntime(cfg); ok {
			target = "qwim-mmh3"
		}
	}
	workCtx, cancel := context.WithCancel(ctx)
	release, err := s.runtime.AcquireWorkload(workCtx, bundle, target, cfg.Runtime.MemoryReserveGiB, peak...)
	if err != nil {
		cancel()
		return nil, err
	}
	return func(results ...error) error {
		for _, result := range results {
			if result != nil {
				cancel()
			}
		}
		defer cancel()
		return release()
	}, nil
}

// Keep extraction and inference as separate leases: the ASR allocation can be
// decided only after decoding, rather than guessing from a compressed upload.
func (s *Server) prepareASRWorkload(ctx context.Context, client *asr.Client) (*asr.Client, func(...error) error, error) {
	if client == nil {
		return nil, nil, fmt.Errorf("ASR client unavailable")
	}
	cfg, _ := s.snapshot()
	if s.runtime == nil || cfg.Runtime.Mode != "managed" {
		return client, func(...error) error { return nil }, nil
	}
	bundleID := cfg.Runtime.ActiveBundle
	if bundleID == "" {
		bundleID = cfg.Runtime.Bundle
	}
	bundle, ok := s.runtime.Catalog().Bundle(bundleID)
	if !ok || !bundle.WorkloadSwap {
		return client, func(...error) error { return nil }, nil
	}
	release, err := s.acquireWorkload(ctx, "extra-media")
	if err != nil {
		return nil, nil, err
	}
	admitted := client.WithAdmission(func(ctx context.Context, audioBytes int64, diarize bool) error {
		if err := release(); err != nil {
			return err
		}
		var peak []float64
		cfg, _ := s.snapshot()
		if s.runtime != nil {
			bundle := cfg.Runtime.ActiveBundle
			if bundle == "" {
				bundle = cfg.Runtime.Bundle
			}
			component, ok := s.runtime.Catalog().ResolveComponent(bundle, "nemotron-asr")
			if ok && component.ComposeAsset == "compose.nemotron-asr.yaml" && component.Model == "nemotron-3.5-asr-streaming-0.6b" {
				budget := nemotronRequestBudget(audioBytes, diarize)
				// Preserve deliberately larger user budgets.
				if component.MemoryGiB > 6 {
					budget = math.Max(budget, component.MemoryGiB)
				}
				peak = []float64{budget}
			}
		}
		next, err := s.acquireWorkload(ctx, "nemotron-asr", peak...)
		if err != nil {
			return err
		}
		release = next
		return nil
	})
	return admitted, func(result ...error) error { return release(result...) }, nil
}

// Retain the qualified Q8 RNNT + streaming diarization workspace allowance
// after the Q5_K weight migration. Quantization savings are observed residency,
// not a reason to reduce unmeasured long-input workspace peaks. Input is PCM16,
// not compressed upload bytes. Retain an explicit length-dependent allowance
// for full-upload copies, float samples and feature/recognition workspaces.
func nemotronRequestBudget(audioBytes int64, diarize bool) float64 {
	budget := 3.75 + float64(max(0, audioBytes))*32/(1<<30)
	if diarize {
		budget += .25
	}
	return math.Ceil(budget*4) / 4
}

func finishWorkload(release func(...error) error, result *error) {
	if err := release(*result); err != nil {
		if *result == nil {
			*result = fmt.Errorf("작업 서비스 메모리 회수 실패: %w", err)
		} else {
			*result = fmt.Errorf("%w; 메모리 회수 실패: %v", *result, err)
		}
	}
}
