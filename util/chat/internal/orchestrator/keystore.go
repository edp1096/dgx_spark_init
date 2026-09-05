package orchestrator

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os/user"
	"path/filepath"
	"sort"
	"time"
)

const keyStoreImage = "sparktalk-extra-ssh:latest"

type KeyManifest struct {
	Schema     int                        `json:"schema"`
	Repository string                     `json:"repository"`
	Authority  string                     `json:"authority"`
	Epoch      uint64                     `json:"epoch"`
	Version    uint64                     `json:"version"`
	Keys       map[string]json.RawMessage `json:"keys"`
	KnownHosts string                     `json:"known_hosts_hash"`
}
type KeyReplica struct {
	Host     string      `json:"host"`
	Node     string      `json:"node"`
	Manifest KeyManifest `json:"manifest"`
	Error    string      `json:"error,omitempty"`
}
type KeyStoreReport struct {
	Replicas      []KeyReplica `json:"replicas"`
	AuthorityHost string       `json:"authority_host"`
	Error         string       `json:"error,omitempty"`
}

func (c *Controller) keyCommand(ctx context.Context, hostID, action string, input []byte) ([]byte, error) {
	host, ok := c.keyStorePeers[hostID]
	if !ok {
		host, ok = c.Catalog().Hosts[hostID]
	}
	if isLocalKeyHost(host.Address) {
		host.Address = ""
	}
	if !ok {
		return nil, fmt.Errorf("unknown key store host %s", hostID)
	}
	directory := host.DataDir
	if host.Address == "" && directory == "" {
		c.mu.RLock()
		directory = c.dataDir
		c.mu.RUnlock()
	}
	if directory == "" || !filepath.IsAbs(directory) {
		return nil, fmt.Errorf("%s: absolute data_dir required", hostID)
	}
	ctx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	// Run a network-isolated helper even when the SSH API container is stopped.
	// Keys never pass through the plaintext Extra HTTP API.
	network := "none"
	if action == "trust" {
		network = "host"
	}
	args := []string{"sh", "-c", `set -eu; base="$1"; network="$2"; shift 2; mkdir -p "$base/extra/ssh/keys" "$base/extra/ssh/state"; exec docker run --rm -i --network "$network" --user "$(id -u):$(id -g)" -v "$base/extra/ssh/keys:/run/sparktalk-extra/keys" -v "$base/extra/ssh/state:/var/lib/sparktalk-extra" "$@"`, "sh", directory, network, keyStoreImage, "-key-store", action}
	cmd := hostCommand(ctx, host, args...)
	cmd.Stdin = bytes.NewReader(input)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("%s key store %s: %s: %w", hostID, action, stderr.String(), err)
	}
	return stdout.Bytes(), nil
}
func (c *Controller) keyReport(ctx context.Context, hosts []string) KeyStoreReport {
	report := KeyStoreReport{Replicas: make([]KeyReplica, len(hosts))}
	// Always consult every configured replica. Offline replicas remain visible.
	for i, host := range hosts {
		r := KeyReplica{Host: host}
		out, err := c.keyCommand(ctx, host, "status", nil)
		if err == nil {
			err = json.Unmarshal(out, &r)
		}
		if err != nil {
			r.Error = err.Error()
		}
		r.Host = host
		report.Replicas[i] = r
	}
	return resolveKeyAuthority(report)
}

func resolveKeyAuthority(report KeyStoreReport) KeyStoreReport {
	nodes := map[string]bool{}
	for _, r := range report.Replicas {
		if r.Error != "" {
			continue
		}
		if nodes[r.Node] {
			report.Error = "복제 호스트의 노드 ID가 중복됐습니다. 저장소 폴더 전체를 복사하지 마세요."
			return report
		}
		nodes[r.Node] = true
	}

	var latest *KeyReplica
	for i := range report.Replicas {
		r := &report.Replicas[i]
		if r.Error != "" || r.Manifest.Repository == "" {
			continue
		}
		if latest != nil && r.Manifest.Repository != latest.Manifest.Repository {
			report.Error = "서로 다른 키 저장소가 있습니다. 자동 병합하지 않습니다."
			return report
		}
		if latest == nil || r.Manifest.Epoch > latest.Manifest.Epoch || (r.Manifest.Epoch == latest.Manifest.Epoch && r.Manifest.Version > latest.Manifest.Version) {
			latest = r
		}
	}
	if latest == nil {
		for _, r := range report.Replicas {
			if r.Error != "" {
				report.Error = "초기화 전에 모든 키 저장소 호스트에 연결해야 합니다"
				return report
			}
		}
		for _, r := range report.Replicas {
			if r.Error == "" {
				report.AuthorityHost = r.Host
				break
			}
		}
		return report
	}
	for _, r := range report.Replicas {
		if r.Error == "" && r.Manifest.Repository != "" && r.Manifest.Epoch == latest.Manifest.Epoch && r.Manifest.Version == latest.Manifest.Version {
			a, _ := json.Marshal(r.Manifest)
			b, _ := json.Marshal(latest.Manifest)
			if !bytes.Equal(a, b) {
				report.Error = "키 저장소 버전 충돌: 자동 덮어쓰기를 중단했습니다."
				return report
			}
		}
	}
	for _, r := range report.Replicas {
		if r.Error == "" && r.Node == latest.Manifest.Authority {
			report.AuthorityHost = r.Host
			return report
		}
	}
	report.Error = "키 관리 호스트가 오프라인이거나 등록되지 않았습니다. 복제본을 최신으로 간주하지 않습니다."
	return report
}
func (c *Controller) syncKeys(ctx context.Context, hosts []string) (KeyStoreReport, error) {
	report := c.keyReport(ctx, hosts)
	if report.Error != "" {
		return report, errors.New(report.Error)
	}
	if report.AuthorityHost == "" {
		return report, errors.New("연결 가능한 키 저장소가 없습니다")
	}
	// A handoff may have fenced the old authority before the target received the
	// commit. Recover by applying the highest committed manifest to the new owner.
	var latest *KeyReplica
	for i := range report.Replicas {
		r := &report.Replicas[i]
		if r.Error == "" && (latest == nil || r.Manifest.Epoch > latest.Manifest.Epoch || (r.Manifest.Epoch == latest.Manifest.Epoch && r.Manifest.Version > latest.Manifest.Version)) {
			latest = r
		}
	}
	if latest == nil || latest.Manifest.Repository == "" {
		return report, nil
	}
	var archive []byte
	for i := range report.Replicas {
		r := &report.Replicas[i]
		if r.Error != "" {
			continue
		}
		a, _ := json.Marshal(r.Manifest)
		b, _ := json.Marshal(latest.Manifest)
		if bytes.Equal(a, b) {
			continue
		}
		if archive == nil {
			var err error
			archive, err = c.keyCommand(ctx, latest.Host, "export", nil)
			if err != nil {
				return report, err
			}
		}
		out, err := c.keyCommand(ctx, r.Host, "apply", archive)
		if err != nil {
			r.Error = err.Error()
			continue
		}
		host := r.Host
		if err = json.Unmarshal(out, r); err != nil {
			return report, err
		}
		r.Host = host
	}
	return report, nil
}

// KeyStore serializes each app's operations; the store's flock and authority
// generation also fence concurrent operations from other app instances.
func (c *Controller) KeyStore(ctx context.Context, hosts []string, peers map[string]Host, action, target string, input []byte) (json.RawMessage, KeyStoreReport, error) {
	c.keyStoreMu.Lock()
	defer c.keyStoreMu.Unlock()
	c.keyStorePeers = peers
	if len(hosts) == 0 {
		return nil, KeyStoreReport{}, errors.New("키 동기화 호스트가 설정되지 않았습니다")
	}
	if action == "status" {
		r := c.keyReport(ctx, hosts)
		return nil, r, nil
	}
	report, err := c.syncKeys(ctx, hosts)
	if err != nil {
		return nil, report, err
	}
	if action == "sync" {
		return nil, report, nil
	}
	if action == "handoff" {
		var node string
		for _, r := range report.Replicas {
			if r.Host == target && r.Error == "" {
				node = r.Node
			}
		}
		if node == "" {
			return nil, report, errors.New("권한 이전 대상에 먼저 동기화해야 합니다")
		}
		if target == report.AuthorityHost {
			return nil, report, nil
		}
		input, _ = json.Marshal(map[string]string{"target": node})
	}
	switch action {
	case "list", "generate", "import", "replace", "delete", "handoff", "trust":
	default:
		return nil, report, errors.New("invalid key store action")
	}
	out, err := c.keyCommand(ctx, report.AuthorityHost, action, input)
	if err != nil {
		return nil, report, err
	}
	if action != "list" {
		next, syncErr := c.syncKeys(ctx, hosts)
		report = next
		if syncErr != nil {
			report.Error = syncErr.Error()
		}
	}
	return out, report, nil
}
func ValidateKeyStoreHosts(catalog Catalog, hosts []string) error {
	seen := map[string]bool{}
	for _, id := range hosts {
		if _, ok := catalog.Hosts[id]; !ok || seen[id] {
			return fmt.Errorf("invalid or duplicate key store host %q", id)
		}
		seen[id] = true
	}
	if len(hosts) > 8 {
		return errors.New("at most 8 key store hosts")
	}
	return nil
}
func SortedKeyHosts(catalog Catalog) []string {
	ids := make([]string, 0, len(catalog.Hosts))
	for id := range catalog.Hosts {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// Freeze physical connection details independently of the local/worker roles.
// Copying the app config to another main machine must not retarget its keys.
func (c *Controller) PrepareKeyStorePeers(hosts []string, existing map[string]Host) (map[string]Host, error) {
	peers := map[string]Host{}
	for id, host := range existing {
		peers[id] = host
	}
	catalog := c.Catalog()
	for _, id := range hosts {
		if _, ok := peers[id]; ok {
			continue
		}
		host, ok := catalog.Hosts[id]
		if !ok {
			return nil, fmt.Errorf("unknown host %s", id)
		}
		if host.Address == "" {
			destination := "8.8.8.8:53"
			for _, other := range hosts {
				if remote := catalog.Hosts[other]; remote.Address != "" {
					destination = net.JoinHostPort(remote.Address, "22")
					break
				}
			}
			// UDP dial performs a route lookup; no payload is sent.
			conn, err := net.Dial("udp", destination)
			if err != nil {
				return nil, fmt.Errorf("resolve local key store address: %w", err)
			}
			host.Address = conn.LocalAddr().(*net.UDPAddr).IP.String()
			conn.Close()
			if host.User == "" {
				u, err := user.Current()
				if err != nil {
					return nil, err
				}
				host.User = u.Username
			}
			if host.DataDir == "" {
				c.mu.RLock()
				host.DataDir = c.dataDir
				c.mu.RUnlock()
			}
		}
		peers[id] = host
	}
	return peers, nil
}
func isLocalKeyHost(address string) bool {
	ip := net.ParseIP(address)
	if ip == nil {
		return false
	}
	if ip.IsLoopback() {
		return true
	}
	addrs, _ := net.InterfaceAddrs()
	for _, addr := range addrs {
		local, _, err := net.ParseCIDR(addr.String())
		if err == nil && local.Equal(ip) {
			return true
		}
	}
	return false
}
