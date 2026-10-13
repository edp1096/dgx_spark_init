package sourcebook

import (
	"bytes"
	"fmt"
	"golang.org/x/net/html"
	"net/url"
	"path"
	"sort"
	"strings"
)

// HTML bundles have no EPUB spine. Prefer their index/contents link order;
// explicitly report documents not reachable from that order.
func htmlBundleOrder(names []string, read func(string) ([]byte, error)) ([]string, bool, error) {
	sort.Strings(names)
	if len(names) < 2 {
		return names, true, nil
	}
	known := map[string]bool{}
	for _, name := range names {
		known[name] = true
	}
	root := ""
	for _, name := range names {
		base := strings.ToLower(path.Base(name))
		if base == "index.html" || base == "index.htm" || base == "contents.html" || base == "toc.html" {
			if root == "" || len(name) < len(root) {
				root = name
			}
		}
	}
	if root == "" {
		return names, false, nil
	}
	seen := map[string]bool{}
	order := []string{}
	var visit func(string) error
	visit = func(name string) error {
		if seen[name] {
			return nil
		}
		seen[name] = true
		order = append(order, name)
		data, e := read(name)
		if e != nil {
			return e
		}
		doc, e := html.Parse(bytes.NewReader(data))
		if e != nil {
			return e
		}
		targets := []string{}
		nodes := 0
		var links func(*html.Node, int) error
		links = func(n *html.Node, depth int) error {
			nodes++
			if depth > 256 || nodes > 250000 {
				return fmt.Errorf("HTML structure exceeds limits")
			}
			if n.Type == html.ElementNode && n.Data == "a" {
				href := attr(n, "href")
				u, e := url.Parse(href)
				if e == nil && u.Scheme == "" && u.Host == "" && u.Path != "" {
					value, e := url.PathUnescape(u.Path)
					if e == nil {
						target := path.Clean(path.Join(path.Dir(name), value))
						if known[target] {
							targets = append(targets, target)
						}
					}
				}
			}
			for c := n.FirstChild; c != nil; c = c.NextSibling {
				if e := links(c, depth+1); e != nil {
					return e
				}
			}
			return nil
		}
		if e = links(doc, 0); e != nil {
			return e
		}
		for _, target := range targets {
			if e = visit(target); e != nil {
				return e
			}
		}
		return nil
	}
	if e := visit(root); e != nil {
		return nil, false, e
	}
	complete := len(order) == len(names)
	for _, name := range names {
		if !seen[name] {
			order = append(order, name)
		}
	}
	return order, complete, nil
}
