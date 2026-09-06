package orchestrator

import (
	"reflect"
	"testing"
)

func fixtureNode(id, lan, rail string) NetworkNode {
	return NetworkNode{ID: id, Hostname: "spark-" + id, Address: lan, Home: "/home/test", Interfaces: []NetworkInterface{
		{Name: "enP7s7", Address: lan, Prefix: 24},
		{Name: "enp1s0f1np1", Address: rail, Prefix: 24, HCA: "rocep1s0f1"},
	}}
}

func TestNetworkRoleReversal(t *testing.T) {
	catalog, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	first := fixtureNode("first", "192.168.100.60", "10.200.0.2")
	second := fixtureNode("second", "192.168.100.61", "10.200.0.1")
	mapped, err := MapNetwork(catalog, second, first)
	if err != nil {
		t.Fatal(err)
	}
	reversed, err := MapNetwork(mapped, first, second)
	if err != nil {
		t.Fatal(err)
	}
	if reversed.Hosts["local"].Address != "" || reversed.Hosts["worker"].Address != second.Address {
		t.Fatal("incorrect host roles")
	}
	found := 0
	for _, b := range reversed.Bundles {
		for _, c := range reversed.BundleComponents(b.ID) {
			if c.Host == "worker" && c.AutoAddress {
				if c.BindAddress != second.Address {
					t.Errorf("%s not remapped: %s", c.ID, c.BindAddress)
				}
			}
			if c.isCluster() {
				found++
				if c.RuntimeOptions["HEAD_RAIL_IP"] != "10.200.0.2" || c.RuntimeOptions["WORKER_RAIL_IP"] != "10.200.0.1" {
					t.Errorf("%s has stale QSFP roles: %v", c.ID, c.RuntimeOptions)
				}
			}
		}
	}
	if found < 2 {
		t.Fatal("expected GLM and DeepSeek")
	}
	for _, c := range reversed.Components {
		if c.isCluster() && c.RuntimeOptions["HEAD_RAIL_IP"] != "10.200.0.2" {
			t.Fatal("preparation definition has stale head")
		}
	}
	if reversed.Network.WorkerID != second.ID || len(reversed.Network.Nodes) != 2 {
		t.Fatal("physical node identities lost")
	}
	again, err := MapNetwork(reversed, first, second)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(reversed, again) {
		t.Fatal("mapping must be idempotent")
	}
}

func TestNetworkPreservesManualAndExternalServices(t *testing.T) {
	catalog, _ := LoadCatalog()
	for i := range catalog.Components {
		catalog.Components[i].AutoAddress = false
	}
	before, _ := ValidateCatalog(catalog)
	mapped, err := MapNetwork(before, fixtureNode("a", "192.168.100.60", "10.200.0.2"), fixtureNode("b", "192.168.100.61", "10.200.0.1"))
	if err != nil {
		t.Fatal(err)
	}
	for _, b := range before.Bundles {
		if !reflect.DeepEqual(before.BundleComponents(b.ID), mapped.BundleComponents(b.ID)) {
			t.Fatalf("manual settings changed in %s", b.ID)
		}
	}
	catalog.Components[0].Controller = "external"
	catalog.Components[0].AutoAddress = true
	catalog.Components[0].Endpoint = "https://third.example:8443/v1"
	catalog.Components[0].HealthURL = "https://third.example:8443/health"
	mapped, err = MapNetwork(catalog, fixtureNode("a", "192.168.100.60", "10.200.0.2"), fixtureNode("b", "192.168.100.61", "10.200.0.1"))
	if err != nil {
		t.Fatal(err)
	}
	if mapped.Components[0].Endpoint != "https://third.example:8443/v1" {
		t.Fatal("external endpoint changed")
	}
}

func TestNetworkRejectsSelfAndMissingRail(t *testing.T) {
	catalog, _ := LoadCatalog()
	a := fixtureNode("a", "192.168.100.60", "10.200.0.2")
	if _, err := MapNetwork(catalog, a, a); err == nil {
		t.Fatal("accepted self as worker")
	}
	b := fixtureNode("b", "192.168.100.61", "10.201.0.1")
	mapped, err := MapNetwork(catalog, a, b)
	if err != nil {
		t.Fatal(err)
	}
	controller, _ := NewControllerWithCatalog(mapped)
	for _, c := range mapped.Components {
		if c.isCluster() && controller.CheckAutoCluster(c) == nil {
			t.Fatal("missing QSFP must prevent startup")
		}
	}
}

func TestAvahiCandidateFiltering(t *testing.T) {
	raw := "=;enP7s7;IPv4;s;SSH;local;spark-one.local;192.168.100.60;22;\n=;enP7s7;IPv4;s;SSH;local;spark-one.local;192.168.100.60;22;\n=;br-test;IPv4;s;SSH;local;spark-one.local;172.17.0.1;22;\n=;enP7s7;IPv6;s;SSH;local;spark-one.local;fe80::1;22;"
	if got := avahiCandidates(raw); !reflect.DeepEqual(got, []string{"192.168.100.60"}) {
		t.Fatal(got)
	}
}

func TestPlanUnaddressedRails(t *testing.T) {
	a := fixtureNode("a", "192.168.100.60", "")
	b := fixtureNode("b", "192.168.100.61", "")
	options := railOptions(a, b)
	if options["HEAD_RAIL_IP"] != "10.200.0.1" || options["WORKER_RAIL_IP"] != "10.200.0.2" {
		t.Fatal(options)
	}
	a.Interfaces[1].Address = "10.200.0.2"
	options = railOptions(a, b)
	if options["HEAD_RAIL_IP"] != "10.200.0.2" || options["WORKER_RAIL_IP"] != "10.200.0.1" {
		t.Fatal("must preserve existing head IP", options)
	}
	a.Interfaces[1].Address = ""
	a.Interfaces[0].Address = "10.200.0.60"
	options = railOptions(a, b)
	if options["NCCL_SUBNET"] == "10.200.0.0/24" {
		t.Fatal("must avoid LAN subnet")
	}
}
