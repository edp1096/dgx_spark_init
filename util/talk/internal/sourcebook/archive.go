package sourcebook

import (
	"archive/zip"
	"bytes"
	"fmt"
	"io"
	"mime"
	"net/url"
	"path"
	"strings"
)

func (b *Book) archive(data []byte, external Loader, epub bool) error {
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
		if f.UncompressedSize64 > (128<<20)-expanded {
			return fmt.Errorf("archive expands beyond limit")
		}
		expanded += f.UncompressedSize64
		if !f.FileInfo().IsDir() {
			if files[name] != nil {
				return fmt.Errorf("duplicate archive path")
			}
			files[name] = f
		}
	}
	encrypted := map[string]bool{}
	read := func(name string) ([]byte, error) {
		f := files[path.Clean(name)]
		if f == nil {
			return nil, fmt.Errorf("archive entry missing: %s", name)
		}
		if encrypted[path.Clean(name)] || f.Flags&1 != 0 {
			return nil, fmt.Errorf("encrypted EPUB resource %q cannot be read; an unencrypted copy is required", name)
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
	var docs []archiveDocument
	if epub || files["META-INF/container.xml"] != nil {
		if files["META-INF/encryption.xml"] != nil {
			raw, err := read("META-INF/encryption.xml")
			if err != nil {
				return err
			}
			encrypted, err = epubEncryptedPaths(raw)
			if err != nil {
				return err
			}
		}
		docs, e = b.epubDocuments(read)
		if e != nil {
			return e
		}
	} else {
		var order []string
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
		for _, name := range ordered {
			docs = append(docs, archiveDocument{Name: name, MIME: "application/xhtml+xml"})
		}
		if !complete {
			b.Warnings = append(b.Warnings, "HTML ZIP reading order is not fully specified by its index; unmatched documents require review")
		}
	}
	if len(docs) == 0 {
		return fmt.Errorf("archive has no HTML documents")
	}
	prefixes := map[string]string{}
	for i, doc := range docs {
		prefix := fmt.Sprintf("d%d-", i+1)
		prefixes[doc.Name] = prefix
		for _, alias := range doc.Aliases {
			prefixes[alias] = prefix
		}
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
		name := strings.TrimPrefix(u.Path, "/")
		v, e := read(name)
		return v, mime.TypeByExtension(path.Ext(name)), e
	}
	start := len(b.Root.Children)
	for _, doc := range docs {
		name := doc.Name
		raw, e := read(name)
		if e != nil {
			return e
		}
		u := url.URL{Scheme: "https", Host: "archive.invalid", Path: "/" + name}
		if e = b.html(raw, u.String(), prefixes[name], doc.MIME+"; charset=utf-8", load); e != nil {
			return e
		}
	}
	var links func(*Node)
	links = func(n *Node) {
		if n.Tag == "a" {
			u, e := url.Parse(n.Attr["href"])
			if e == nil && u.Host == "archive.invalid" {
				name := strings.TrimPrefix(u.Path, "/")
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
