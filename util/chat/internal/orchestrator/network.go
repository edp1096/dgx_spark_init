package orchestrator

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"net"
	"net/url"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"time"
)

// Roles belong to this app instance, while Nodes remember physical machines.
type AutoNetwork struct {
	Enabled   bool          `json:"enabled" yaml:"enabled"`
	HeadID    string        `json:"head_id,omitempty" yaml:"head_id,omitempty"`
	WorkerID  string        `json:"worker_id,omitempty" yaml:"worker_id,omitempty"`
	Nodes     []NetworkNode `json:"nodes,omitempty" yaml:"nodes,omitempty"`
	LastError string        `json:"last_error,omitempty" yaml:"last_error,omitempty"`
}

type NetworkInterface struct {
	Name    string `json:"name" yaml:"name"`
	Address string `json:"address" yaml:"address"`
	Prefix  int    `json:"prefix" yaml:"prefix"`
	HCA     string `json:"hca,omitempty" yaml:"hca,omitempty"`
}

type NetworkNode struct {
	System     string             `json:"system,omitempty" yaml:"system,omitempty"`
	ID         string             `json:"id" yaml:"id"`
	Hostname   string             `json:"hostname" yaml:"hostname"`
	Address    string             `json:"address" yaml:"address"`
	Home       string             `json:"home" yaml:"home"`
	Interfaces []NetworkInterface `json:"interfaces" yaml:"interfaces"`
}

type NetworkDiscovery struct {
	Local      NetworkNode   `json:"local"`
	Candidates []NetworkNode `json:"candidates"`
	Warnings   []string      `json:"warnings,omitempty"`
	Catalog    *Catalog      `json:"catalog,omitempty"`
	Error      string        `json:"error,omitempty"`
}

// Fixed read-only probe. No keys, sudo, IP changes, or downloaded scripts.
const networkProbe = `import json,socket,os,pathlib,subprocess
interfaces=[]
for dev in json.loads(subprocess.check_output(['ip','-j','address','show','up'])):
 name=dev['ifname']
 if name=='lo' or name.startswith(('docker','br-','veth')): continue
 hcas=sorted(p.name for p in pathlib.Path('/sys/class/net',name,'device/infiniband').glob('*'))
 found=False
 for a in dev.get('addr_info',[]):
  if a.get('family')=='inet':
   found=True
   interfaces.append(dict(name=name,address=a['local'],prefix=a['prefixlen'],hca=hcas[0] if hcas else ''))
 if not found and hcas and 'LOWER_UP' in dev.get('flags',[]): interfaces.append(dict(name=name,address='',prefix=0,hca=hcas[0]))
product=pathlib.Path('/sys/class/dmi/id/product_name')
print(json.dumps(dict(id=pathlib.Path('/etc/machine-id').read_text().strip(),hostname=socket.gethostname(),system=product.read_text().strip() if product.exists() else '',home=os.path.expanduser('~'),address='',interfaces=interfaces)))
`

func inspectNetworkNode(ctx context.Context, host Host) (NetworkNode, error) {
	ctx, cancel := context.WithTimeout(ctx, 8*time.Second)
	defer cancel()
	raw, err := executeHost(ctx, host, []byte(networkProbe), "python3", "-")
	if err != nil {
		return NetworkNode{}, fmt.Errorf("%s SSH/네트워크 확인 실패: %w", host.Address, err)
	}
	var node NetworkNode
	if err := json.Unmarshal(raw, &node); err != nil {
		return node, err
	}
	if node.ID == "" || len(node.Interfaces) == 0 {
		return node, fmt.Errorf("장비 ID 또는 IPv4 주소를 확인할 수 없습니다")
	}
	node.Address = host.Address
	return node, nil
}

func nodeOwns(node NetworkNode, address string) bool {
	address = strings.TrimSuffix(address, ".local")
	if address == node.Hostname || address == "localhost" {
		return true
	}
	for _, i := range node.Interfaces {
		if address == i.Address {
			return true
		}
	}
	ip := net.ParseIP(address)
	return ip != nil && ip.IsLoopback()
}

func avahiCandidates(raw string) []string {
	seen := map[string]bool{}
	var out []string
	for _, line := range strings.Split(raw, "\n") {
		f := strings.Split(line, ";")
		if len(f) < 9 || f[0] != "=" || f[2] != "IPv4" || !strings.HasPrefix(f[6], "spark-") {
			continue
		}
		if strings.HasPrefix(f[1], "br-") || strings.HasPrefix(f[1], "veth") || f[1] == "lo" {
			continue
		}
		ip := net.ParseIP(f[7])
		if ip == nil || ip.IsLoopback() || !ip.IsGlobalUnicast() {
			continue
		}
		if !seen[f[7]] {
			out = append(out, f[7])
			seen[f[7]] = true
		}
	}
	return out
}

func DiscoverNetwork(ctx context.Context, source Catalog, choice string) NetworkDiscovery {
	ctx, cancelAll := context.WithTimeout(ctx, 25*time.Second)
	defer cancelAll()
	result := NetworkDiscovery{Candidates: []NetworkNode{}}
	catalog, err := ValidateCatalog(source)
	if err != nil {
		result.Error = err.Error()
		return result
	}
	local, err := inspectNetworkNode(ctx, Host{})
	if err != nil {
		result.Error = err.Error()
		return result
	}
	result.Local = local
	worker, exists := catalog.Hosts["worker"]
	if !exists {
		result.Error = "worker 실행호스트를 등록하세요"
		return result
	}
	addresses := []string{worker.Address}
	registered := map[string]bool{worker.Address: true}
	if catalog.Network != nil {
		for _, node := range catalog.Network.Nodes {
			if node.ID != local.ID {
				addresses = append(addresses, node.Address, node.Hostname+".local")
				registered[node.Address] = true
				registered[node.Hostname+".local"] = true
			}
		}
	}
	scanCtx, cancel := context.WithTimeout(ctx, 4*time.Second)
	raw, _ := exec.CommandContext(scanCtx, "avahi-browse", "-p", "-r", "-t", "_ssh._tcp").Output()
	cancel()
	addresses = append(addresses, avahiCandidates(string(raw))...)
	// An already trusted SSH neighbour is a useful fallback when mDNS IPv4
	// advertisements are absent. This does not scan a subnet or trust new keys.
	neighborRaw, _ := exec.CommandContext(ctx, "ip", "-j", "-4", "neigh", "show").Output()
	var neighbors []struct {
		Address string `json:"dst"`
		Device  string `json:"dev"`
	}
	_ = json.Unmarshal(neighborRaw, &neighbors)
	for _, neighbor := range neighbors {
		physical := false
		for _, i := range local.Interfaces {
			physical = physical || i.Name == neighbor.Device
		}
		if physical && net.ParseIP(neighbor.Address) != nil {
			if exec.CommandContext(ctx, "ssh-keygen", "-F", neighbor.Address).Run() == nil {
				addresses = append(addresses, neighbor.Address)
			}
		}
	}
	seenAddress, seenID := map[string]bool{}, map[string]bool{}
	for _, address := range addresses {
		if address == "" || seenAddress[address] || nodeOwns(local, address) {
			continue
		}
		seenAddress[address] = true
		if len(seenAddress) > 16 {
			break
		}
		host := worker
		host.Address = address
		peer, e := inspectNetworkNode(ctx, host)
		if e != nil {
			result.Warnings = append(result.Warnings, address+": SSH 확인 실패 (기존 SSH 인증과 known_hosts 필요)")
			continue
		}
		if peer.ID == local.ID || seenID[peer.ID] {
			continue
		}
		product := strings.ToLower(peer.System)
		if !registered[address] && !strings.Contains(product, "spark") && !strings.Contains(product, "gx10") {
			continue
		}
		// Use a routed management address, not a QSFP/mDNS alias, for APIs and SSH.
		for _, pi := range peer.Interfaces {
			if pi.HCA != "" {
				continue
			}
			for _, li := range local.Interfaces {
				_, subnet, _ := net.ParseCIDR(fmt.Sprintf("%s/%d", li.Address, li.Prefix))
				if li.HCA == "" && subnet != nil && subnet.Contains(net.ParseIP(pi.Address)) {
					peer.Address = pi.Address
				}
			}
		}
		if peer.Address != address {
			host.Address = peer.Address
			check, e := inspectNetworkNode(ctx, host)
			if e != nil || check.ID != peer.ID {
				result.Warnings = append(result.Warnings, peer.Hostname+": 관리망 SSH 주소 확인 실패")
				continue
			}
		}
		seenID[peer.ID] = true
		result.Candidates = append(result.Candidates, peer)
	}
	selected := choice
	if selected == "" && catalog.Network != nil {
		selected = catalog.Network.WorkerID
	}
	if selected == local.ID {
		selected = ""
	} // old worker is now the app/head
	var peer *NetworkNode
	for i := range result.Candidates {
		if result.Candidates[i].ID == selected {
			peer = &result.Candidates[i]
		}
	}
	if selected != "" && peer == nil {
		result.Error = "등록된 워커를 찾지 못했습니다. 장비를 켜거나 다른 워커를 선택하세요"
		return result
	}
	if peer == nil && len(result.Candidates) == 1 {
		peer = &result.Candidates[0]
	}
	if peer == nil {
		result.Error = "워커를 하나 선택하세요. 탐색되지 않으면 worker의 SSH 주소를 입력하세요"
		return result
	}
	if catalog.Network != nil && catalog.Network.HeadID != "" && catalog.Network.HeadID != local.ID {
		names := map[string]bool{}
		for _, c := range catalog.Components {
			if c.AutoAddress && c.isCluster() {
				names[c.Container] = true
				names[c.WorkerContainer] = true
			}
		}
		for _, host := range []Host{{}, func() Host { h := worker; h.Address = peer.Address; return h }()} {
			for name := range names {
				out, _ := executeHost(ctx, host, nil, "docker", "inspect", "-f", "{{.State.Running}}", name)
				if strings.TrimSpace(string(out)) == "true" {
					result.Error = "헤드 역할이 바뀌었습니다. 이전 헤드의 실행 중인 클러스터 세트를 먼저 중지한 뒤 다시 탐색하세요"
					return result
				}
			}
		}
	}
	conn, e := (&net.Dialer{}).DialContext(ctx, "udp", net.JoinHostPort(peer.Address, "22"))
	if e != nil {
		result.Error = e.Error()
		return result
	}
	local.Address = conn.LocalAddr().(*net.UDPAddr).IP.String()
	conn.Close()
	result.Local = local
	mapped, e := MapNetwork(catalog, local, *peer)
	if e != nil {
		result.Error = e.Error()
		return result
	}
	result.Catalog = &mapped
	return result
}

func mapURL(raw, address string, publishedPort int) string {
	u, err := url.Parse(raw)
	if err != nil || u.Host == "" {
		return raw
	}
	port := u.Port()
	if publishedPort > 0 {
		port = strconv.Itoa(publishedPort)
	}
	if port != "" {
		u.Host = net.JoinHostPort(address, port)
	} else {
		u.Host = address
	}
	return u.String()
}

func railOptions(head, worker NetworkNode) map[string]string {
	for _, h := range head.Interfaces {
		for _, w := range worker.Interfaces {
			if h.HCA == "" || w.HCA == "" || h.Address == w.Address {
				continue
			}
			_, subnet, err := net.ParseCIDR(fmt.Sprintf("%s/%d", h.Address, h.Prefix))
			if err != nil || !subnet.Contains(net.ParseIP(w.Address)) {
				continue
			}
			_, other, err := net.ParseCIDR(fmt.Sprintf("%s/%d", w.Address, w.Prefix))
			if err != nil || !other.Contains(net.ParseIP(h.Address)) {
				continue
			}
			return map[string]string{"HEAD_RAIL_IP": h.Address, "WORKER_RAIL_IP": w.Address, "HEAD_NCCL_IF": h.Name, "WORKER_NCCL_IF": w.Name, "HEAD_NCCL_HCA": h.HCA, "WORKER_NCCL_HCA": w.HCA, "NCCL_SUBNET": subnet.String()}
		}
	}
	// Preserve an existing rail address when just the other node lost its IP.
	for _, h := range head.Interfaces {
		for _, w := range worker.Interfaces {
			if h.HCA == "" || w.HCA == "" || (h.Address == "") == (w.Address == "") {
				continue
			}
			known := h
			if known.Address == "" {
				known = w
			}
			_, subnet, e := net.ParseCIDR(fmt.Sprintf("%s/%d", known.Address, known.Prefix))
			if e != nil || subnet.IP.To4() == nil || known.Prefix > 30 {
				continue
			}
			base := binary.BigEndian.Uint32(subnet.IP.To4())
			mask := binary.BigEndian.Uint32(subnet.Mask)
			for offset := uint32(1); offset < 255 && offset < ^mask; offset++ {
				ip := make(net.IP, 4)
				binary.BigEndian.PutUint32(ip, base+offset)
				address := ip.String()
				used := false
				for _, node := range []NetworkNode{head, worker} {
					for _, i := range node.Interfaces {
						used = used || i.Address == address
					}
				}
				if used {
					continue
				}
				ha, wa := h.Address, w.Address
				if ha == "" {
					ha = address
				} else {
					wa = address
				}
				return map[string]string{"HEAD_RAIL_IP": ha, "WORKER_RAIL_IP": wa, "HEAD_NCCL_IF": h.Name, "WORKER_NCCL_IF": w.Name, "HEAD_NCCL_HCA": h.HCA, "WORKER_NCCL_HCA": w.HCA, "NCCL_SUBNET": subnet.String()}
			}
		}
	}
	// A directly connected port may have no IP until a model starts. Plan an
	// unused private subnet; actual assignment happens only during explicit start.
	for _, h := range head.Interfaces {
		for _, w := range worker.Interfaces {
			if h.HCA == "" || w.HCA == "" || h.Address != "" || w.Address != "" {
				continue
			}
			for octet := 0; octet < 256; octet++ {
				base := fmt.Sprintf("10.200.%d.", octet)
				_, subnet, _ := net.ParseCIDR(base + "0/24")
				overlap := false
				for _, node := range []NetworkNode{head, worker} {
					for _, i := range node.Interfaces {
						if i.Address == "" {
							continue
						}
						_, other, e := net.ParseCIDR(fmt.Sprintf("%s/%d", i.Address, i.Prefix))
						if e == nil && (subnet.Contains(other.IP) || other.Contains(subnet.IP)) {
							overlap = true
						}
					}
				}
				if !overlap {
					return map[string]string{"HEAD_RAIL_IP": base + "1", "WORKER_RAIL_IP": base + "2", "HEAD_NCCL_IF": h.Name, "WORKER_NCCL_IF": w.Name, "HEAD_NCCL_HCA": h.HCA, "WORKER_NCCL_HCA": w.HCA, "NCCL_SUBNET": subnet.String()}
				}
			}
		}
	}
	return nil
}

func (c *Controller) ensureAutoRail(ctx context.Context, component Component) error {
	n := c.Catalog().Network
	if n == nil || !n.Enabled || !component.AutoAddress || component.Controller == "external" || (component.Host != "worker" && component.WorkerHost != "worker") {
		return nil
	}
	if err := c.CheckAutoCluster(component); err != nil {
		return err
	}
	_, subnet, err := net.ParseCIDR(component.RuntimeOptions["NCCL_SUBNET"])
	if err != nil {
		return err
	}
	prefix, _ := subnet.Mask.Size()
	for _, side := range []struct{ host, prefix string }{{component.Host, "HEAD"}, {component.WorkerHost, "WORKER"}} {
		address := component.RuntimeOptions[side.prefix+"_RAIL_IP"]
		iface := component.RuntimeOptions[side.prefix+"_NCCL_IF"]
		if net.ParseIP(address) == nil || !safeInterfaceName(iface) {
			return fmt.Errorf("invalid QSFP address/interface")
		}
		probe, e := inspectNetworkNode(ctx, c.host(side.host))
		if e != nil {
			return e
		}
		found, active := false, false
		for _, i := range probe.Interfaces {
			if i.Name == iface && i.HCA != "" {
				active = true
				found = found || i.Address == address
				if i.Address != "" && i.Address != address {
					return fmt.Errorf("%s의 QSFP 주소가 탐색 이후 변경됐습니다. 다시 탐색하세요", side.host)
				}
			}
		}
		if !active {
			return fmt.Errorf("%s의 QSFP 링크가 연결되지 않았습니다", side.host)
		}
		if !found {
			_, e = executeHost(ctx, c.host(side.host), nil, "docker", "run", "--rm", "--network", "host", "--cap-add", "NET_ADMIN", "alpine:3.22", "ip", "address", "add", fmt.Sprintf("%s/%d", address, prefix), "dev", iface)
			if e != nil {
				return fmt.Errorf("%s QSFP 주소 설정: %w", side.host, e)
			}
		}
	}
	_, err = executeHost(ctx, c.host(component.Host), nil, "ping", "-I", component.RuntimeOptions["HEAD_NCCL_IF"], "-c", "2", "-W", "2", component.RuntimeOptions["WORKER_RAIL_IP"])
	if err != nil {
		return fmt.Errorf("QSFP 워커 통신 확인 실패: %w", err)
	}
	return nil
}

func safeInterfaceName(value string) bool {
	if value == "" || len(value) > 64 {
		return false
	}
	for _, r := range value {
		if !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '_' || r == '.' || r == '-') {
			return false
		}
	}
	return true
}

func MapNetwork(source Catalog, head, worker NetworkNode) (Catalog, error) {
	catalog, err := ValidateCatalog(source)
	if err != nil {
		return catalog, err
	}
	if head.ID == worker.ID || head.ID == "" || worker.ID == "" || nodeOwns(head, worker.Address) {
		return catalog, fmt.Errorf("헤드와 워커는 서로 다른 장비여야 합니다")
	}
	h, ok := catalog.Hosts["local"]
	if !ok {
		return catalog, fmt.Errorf("local 실행호스트가 필요합니다")
	}
	h.Address = ""
	catalog.Hosts["local"] = h
	w := catalog.Hosts["worker"]
	w.Address = worker.Address
	catalog.Hosts["worker"] = w
	network := catalog.Network
	if network == nil {
		network = &AutoNetwork{}
	}
	network.Enabled = true
	network.HeadID = head.ID
	network.WorkerID = worker.ID
	network.LastError = ""
	nodes := map[string]NetworkNode{}
	for _, n := range network.Nodes {
		nodes[n.ID] = n
	}
	nodes[head.ID] = head
	nodes[worker.ID] = worker
	network.Nodes = nil
	for _, n := range nodes {
		network.Nodes = append(network.Nodes, n)
	}
	sort.Slice(network.Nodes, func(i, j int) bool { return network.Nodes[i].ID < network.Nodes[j].ID })
	catalog.Network = network
	rails := railOptions(head, worker)
	remap := func(c Component) Component {
		if !c.AutoAddress || c.Controller == "external" {
			return c
		}
		address := ""
		if c.Host == "local" {
			address = "127.0.0.1"
		} else if c.Host == "worker" {
			address = worker.Address
		}
		if address == "" {
			return c
		}
		c.Endpoint = mapURL(c.Endpoint, address, c.Port)
		c.HealthURL = mapURL(c.HealthURL, address, c.Port)
		c.BindAddress = address
		if c.isCluster() && c.Host == "local" && c.WorkerHost == "worker" {
			options := map[string]string{}
			for k, v := range c.RuntimeOptions {
				if k != "HEAD_RAIL_IP" && k != "WORKER_RAIL_IP" {
					options[k] = v
				}
			}
			for k, v := range rails {
				options[k] = v
			}
			c.RuntimeOptions = options
		}
		if c.Port == 0 {
			u, _ := url.Parse(c.Endpoint)
			if u != nil {
				c.Port, _ = strconv.Atoi(u.Port())
			}
		}
		return c
	}
	// Resolve old bindings before changing definitions so overrides keep their meaning.
	effective := map[string][]Component{}
	for _, b := range catalog.Bundles {
		effective[b.ID] = catalog.BundleComponents(b.ID)
	}
	for i, c := range catalog.Components {
		catalog.Components[i] = remap(c)
	}
	for i, b := range catalog.Bundles {
		for _, c := range effective[b.ID] {
			c = remap(c)
			if c.AutoAddress && c.isCluster() && c.Host == "local" && c.WorkerHost == "worker" {
				if rails != nil {
					options := map[string]string{}
					for k, v := range c.RuntimeOptions {
						options[k] = v
					}
					for k, v := range rails {
						options[k] = v
					}
					c.RuntimeOptions = options
				} else {
					options := map[string]string{}
					for k, v := range c.RuntimeOptions {
						if k != "HEAD_RAIL_IP" && k != "WORKER_RAIL_IP" {
							options[k] = v
						}
					}
					c.RuntimeOptions = options
				}
			}
			var base Component
			for _, v := range catalog.Components {
				if v.ID == c.ID {
					base = v
					break
				}
			}
			if catalog.Bundles[i].Bindings == nil {
				catalog.Bundles[i].Bindings = map[string]Deployment{}
			}
			catalog.Bundles[i].Bindings[c.ID] = deploymentDifference(base, c)
		}
	}
	return ValidateCatalog(catalog)
}

func (c *Controller) CheckAutoCluster(component Component) error {
	n := c.Catalog().Network
	if n == nil || !n.Enabled || !component.AutoAddress || !component.isCluster() {
		return nil
	}
	if n.LastError != "" {
		return fmt.Errorf("자동 네트워크 확인 실패: %s", n.LastError)
	}
	if n.HeadID == "" {
		return nil
	} // server resolves the catalog before managed actions
	if component.isCluster() && (component.RuntimeOptions["HEAD_RAIL_IP"] == "" || component.RuntimeOptions["WORKER_RAIL_IP"] == "") {
		return fmt.Errorf("헤드·워커 사이의 QSFP IPv4 연결을 찾지 못했습니다")
	}
	return nil
}
