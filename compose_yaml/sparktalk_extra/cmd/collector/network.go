package main

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"
)

func validatePublicURL(ctx context.Context, rawURL string) (*url.URL, error) {
	parsed, err := url.Parse(strings.TrimSpace(rawURL))
	if err != nil || parsed.Hostname() == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return nil, fmt.Errorf("URL must use public HTTP or HTTPS")
	}
	addresses, err := net.DefaultResolver.LookupIPAddr(ctx, parsed.Hostname())
	if err != nil || len(addresses) == 0 {
		return nil, fmt.Errorf("resolve URL host: %w", err)
	}
	for _, address := range addresses {
		if !publicIP(address.IP) {
			return nil, fmt.Errorf("private or local network URLs are blocked")
		}
	}
	return parsed, nil
}

func publicIP(ip net.IP) bool {
	return ip != nil && !ip.IsLoopback() && !ip.IsPrivate() && !ip.IsLinkLocalUnicast() &&
		!ip.IsLinkLocalMulticast() && !ip.IsMulticast() && !ip.IsUnspecified()
}

func safeHTTPClient(timeout time.Duration) *http.Client {
	dialer := &net.Dialer{Timeout: 15 * time.Second, KeepAlive: 30 * time.Second}
	transport := &http.Transport{
		Proxy: nil,
		DialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
			host, port, err := net.SplitHostPort(address)
			if err != nil {
				return nil, err
			}
			addresses, err := net.DefaultResolver.LookupIPAddr(ctx, host)
			if err != nil {
				return nil, err
			}
			for _, candidate := range addresses {
				if publicIP(candidate.IP) {
					return dialer.DialContext(ctx, network, net.JoinHostPort(candidate.IP.String(), port))
				}
			}
			return nil, fmt.Errorf("private or local network URLs are blocked")
		},
		ForceAttemptHTTP2: true, MaxIdleConns: 20, IdleConnTimeout: 30 * time.Second,
		TLSHandshakeTimeout: 15 * time.Second, ResponseHeaderTimeout: 30 * time.Second,
	}
	return &http.Client{
		Transport: transport, Timeout: timeout,
		CheckRedirect: func(request *http.Request, via []*http.Request) error {
			if len(via) >= 6 {
				return fmt.Errorf("too many redirects")
			}
			_, err := validatePublicURL(request.Context(), request.URL.String())
			return err
		},
	}
}
