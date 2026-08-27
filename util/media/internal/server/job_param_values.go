package server

import (
	"encoding/json"
	"strings"
)

func stringParam(params map[string]any, key, fallback string) string {
	if value, ok := params[key].(string); ok && strings.TrimSpace(value) != "" {
		return value
	}
	return fallback
}

func intParam(params map[string]any, key string, fallback int) int {
	switch value := params[key].(type) {
	case int:
		return value
	case int64:
		return int(value)
	case float64:
		return int(value)
	case json.Number:
		if number, err := value.Int64(); err == nil {
			return int(number)
		}
	}
	return fallback
}

func int64Param(params map[string]any, key string, fallback int64) int64 {
	switch value := params[key].(type) {
	case int:
		return int64(value)
	case int64:
		return value
	case float64:
		return int64(value)
	case json.Number:
		if number, err := value.Int64(); err == nil {
			return number
		}
	}
	return fallback
}

func floatParam(params map[string]any, key string, fallback float64) float64 {
	switch value := params[key].(type) {
	case float64:
		return value
	case float32:
		return float64(value)
	case int:
		return float64(value)
	case int64:
		return float64(value)
	case json.Number:
		if number, err := value.Float64(); err == nil {
			return number
		}
	}
	return fallback
}

func boolParam(params map[string]any, key string, fallback bool) bool {
	if value, ok := params[key].(bool); ok {
		return value
	}
	return fallback
}

func decodeParam(params map[string]any, key string, target any) {
	data, err := json.Marshal(params[key])
	if err == nil {
		_ = json.Unmarshal(data, target)
	}
}
