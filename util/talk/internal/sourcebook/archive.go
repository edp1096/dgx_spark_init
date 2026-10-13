package sourcebook

import (
	"archive/zip"
	"bytes"
	"encoding/xml"
	"fmt"
	"io"
	"mime"
	"net/url"
	"path"
	"strings"
)

func (b *Book) archive(data []byte, external Loader) error {
	z, e := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if e != nil {
		return e
	}
	if len(z.File) > 10000 {
		return fmt.Errorf("too many archive entries")
	}
	files := map[string]*zip.File{}
	var expanded uint64
	for _, f := range z.File {
		name := strings.ReplaceAll(f.Name, "\\", "/")
		if strings.HasPrefix(name, "/") || path.Clean(name) != strings.TrimSuffix(name, "/") || strings.Contains(name, "../") {
			return fmt.Errorf("unsafe archive path %q", name)
		}
		expanded += f.UncompressedSize64
		if expanded > 128<<20 {
			return fmt.Errorf("archive expands beyond limit")
		}
		if !f.FileInfo().IsDir() {
			if files[name] != nil {
				return fmt.Errorf("duplicate archive path")
			}
			files[name] = f
		}
	}
	read := func(name string) ([]byte, error) {
		f := files[path.Clean(name)]
		if f == nil {
			return nil, fmt.Errorf("archive entry missing: %s", name)
		}
		if f.UncompressedSize64 > MaxBytes {
			return nil, fmt.Errorf("archive entry too large")
		}
		r, e := f.Open()
		if e != nil {
			return nil, e
		}
		defer r.Close()
		v, e := io.ReadAll(io.LimitReader(r, MaxBytes+1))
		if len(v) > MaxBytes {
			return nil, fmt.Errorf("archive entry too large")
		}
		return v, e
	}
	order := []string{}
	if files["META-INF/container.xml"] != nil {
		raw, e := read("META-INF/container.xml")
		if e != nil {
			return e
		}
		var container struct {
			Roots []struct {
				Path string `xml:"full-path,attr"`
			} `xml:"rootfiles>rootfile"`
		}
		if e = xml.Unmarshal(raw, &container); e != nil || len(container.Roots) == 0 {
			return fmt.Errorf("invalid EPUB container")
		}
		opf := container.Roots[0].Path
		raw, e = read(opf)
		if e != nil {
			return e
		}
		var pack struct {
			Title    string `xml:"metadata>title"`
			Author   string `xml:"metadata>creator"`
			Language string `xml:"metadata>language"`
			Manifest []struct {
				ID   string `xml:"id,attr"`
				Href string `xml:"href,attr"`
				MIME string `xml:"media-type,attr"`
			} `xml:"manifest>item"`
			Spine []struct {
				ID     string `xml:"idref,attr"`
				Linear string `xml:"linear,attr"`
			} `xml:"spine>itemref"`
		}
		if e = xml.Unmarshal(raw, &pack); e != nil {
			return e
		}
		if pack.Title != "" {
			b.Title = pack.Title
		}
		if b.Metadata == nil {
			b.Metadata = map[string]any{}
		}
		if pack.Author != "" {
			b.Metadata["author"] = pack.Author
		}
		if pack.Language != "" {
			b.Metadata["language"] = pack.Language
		}
		items := map[string]string{}
		for _, it := range pack.Manifest {
			if strings.Contains(it.MIME, "html") {
				u, e := url.PathUnescape(strings.SplitN(it.Href, "#", 2)[0])
				if e != nil {
					return e
				}
				items[it.ID] = path.Clean(path.Join(path.Dir(opf), u))
			}
		}
		for _, it := range pack.Spine {

			name := items[it.ID]
			if name == "" {
				return fmt.Errorf("unsupported EPUB spine item %s", it.ID)
			}
			order = append(order, name)
		}
		if len(order) == 0 {
			return fmt.Errorf("EPUB has no readable spine")
		}
	} else {
		for name := range files {
			ext := strings.ToLower(path.Ext(name))
			if ext == ".html" || ext == ".xhtml" || ext == ".htm" {
				order = append(order, name)
			}
		}
		ordered, complete, err := htmlBundleOrder(order, read)
		if err != nil {
			return err
		}
		order = ordered
		if !complete {
			b.Warnings = append(b.Warnings, "HTML ZIP reading order is not fully specified by its index; unmatched documents require review")
		}
	}
	if len(order) == 0 {
		return fmt.Errorf("archive has no HTML documents")
	}
	prefixes := map[string]string{}
	for i, name := range order {
		prefixes[name] = fmt.Sprintf("d%d-", i+1)
	}
	load := func(target string) ([]byte, string, error) {
		u, e := url.Parse(target)
		if e != nil {
			return nil, "", e
		}
		if u.Host != "archive.invalid" {
			if external != nil {
				return external(target)
			}
			return nil, "", fmt.Errorf("external archive resource unavailable")
		}
		name, e := url.PathUnescape(strings.TrimPrefix(u.Path, "/"))
		if e != nil {
			return nil, "", e
		}
		v, e := read(name)
		return v, mime.TypeByExtension(path.Ext(name)), e
	}
	start := len(b.Root.Children)
	for _, name := range order {
		raw, e := read(name)
		if e != nil {
			return e
		}
		u := url.URL{Scheme: "https", Host: "archive.invalid", Path: "/" + name}
		if e = b.html(raw, u.String(), prefixes[name], "application/xhtml+xml; charset=utf-8", load); e != nil {
			return e
		}
	}
	var links func(*Node)
	links = func(n *Node) {
		if n.Tag == "a" {
			u, e := url.Parse(n.Attr["href"])
			if e == nil && u.Host == "archive.invalid" {
				name, _ := url.PathUnescape(strings.TrimPrefix(u.Path, "/"))
				if p, ok := prefixes[name]; ok {
					n.Attr["href"] = "#" + p + u.Fragment
				} else {
					delete(n.Attr, "href")
				}
			}
		}
		for _, c := range n.Children {
			links(c)
		}
	}
	for _, n := range b.Root.Children[start:] {
		links(n)
	}
	return nil
}
