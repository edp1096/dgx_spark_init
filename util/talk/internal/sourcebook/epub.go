package sourcebook

import (
	"encoding/xml"
	"fmt"
	"net/url"
	"path"
	"strings"
)

type archiveDocument struct {
	Name, MIME string
	Aliases    []string
}

type epubItem struct {
	ID       string `xml:"id,attr"`
	Href     string `xml:"href,attr"`
	MIME     string `xml:"media-type,attr"`
	Fallback string `xml:"fallback,attr"`
}

// URI paths are decoded by url.Parse exactly once. In particular, %2520 is a
// literal %20 in a ZIP filename, not a space.
func archivePath(base, ref string) (string, error) {
	u, err := url.Parse(ref)
	if err != nil || u.IsAbs() || u.Host != "" || strings.HasPrefix(u.Path, "/") || strings.ContainsAny(u.Path, "\\\x00") {
		return "", fmt.Errorf("invalid local archive reference %q", ref)
	}
	name := path.Clean(path.Join(path.Dir(base), u.Path))
	if u.Path == "" || name == "." || name == ".." || strings.HasPrefix(name, "../") {
		return "", fmt.Errorf("archive reference escapes container: %q", ref)
	}
	return name, nil
}

func (b *Book) epubDocuments(read func(string) ([]byte, error)) ([]archiveDocument, error) {
	raw, err := read("META-INF/container.xml")
	if err != nil {
		return nil, fmt.Errorf("read EPUB container: %w", err)
	}
	var container struct {
		Roots []struct {
			Path string `xml:"full-path,attr"`
			MIME string `xml:"media-type,attr"`
		} `xml:"rootfiles>rootfile"`
	}
	if err := xml.Unmarshal(raw, &container); err != nil || len(container.Roots) == 0 {
		return nil, fmt.Errorf("invalid EPUB container")
	}
	opf := ""
	for _, root := range container.Roots {
		if root.MIME == "" || root.MIME == "application/oebps-package+xml" {
			opf, err = archivePath("", root.Path)
			if err != nil {
				return nil, err
			}
			break
		}
	}
	if opf == "" {
		return nil, fmt.Errorf("EPUB container has no package document")
	}
	if len(container.Roots) > 1 {
		b.Warnings = append(b.Warnings, "EPUB has multiple renditions; the default package is used")
	}
	raw, err = read(opf)
	if err != nil {
		return nil, err
	}
	var pack struct {
		Version  string     `xml:"version,attr"`
		Title    string     `xml:"metadata>title"`
		Authors  []string   `xml:"metadata>creator"`
		Language string     `xml:"metadata>language"`
		Manifest []epubItem `xml:"manifest>item"`
		Spine    []struct {
			ID string `xml:"idref,attr"`
		} `xml:"spine>itemref"`
	}
	if err = xml.Unmarshal(raw, &pack); err != nil {
		return nil, fmt.Errorf("invalid EPUB package: %w", err)
	}
	b.Title = strings.TrimSpace(pack.Title)
	b.Metadata = map[string]any{"source_format": "epub", "epub_version": pack.Version}
	if len(pack.Authors) > 0 {
		b.Metadata["author"] = strings.Join(pack.Authors, "; ")
	}
	if pack.Language != "" {
		b.Metadata["language"] = pack.Language
	}
	items := map[string]epubItem{}
	for _, item := range pack.Manifest {
		if item.ID == "" || items[item.ID].ID != "" {
			return nil, fmt.Errorf("EPUB manifest has an empty or duplicate item ID %q", item.ID)
		}
		items[item.ID] = item
	}
	var docs []archiveDocument
	seen := map[string]int{}
	for _, ref := range pack.Spine {
		id := ref.ID
		visited := map[string]bool{}
		var aliases []string
		var doc archiveDocument
		for {
			item, ok := items[id]
			if !ok {
				return nil, fmt.Errorf("EPUB spine/fallback references missing manifest item %q", id)
			}
			if visited[id] {
				return nil, fmt.Errorf("EPUB fallback cycle at %q", id)
			}
			visited[id] = true
			name, err := archivePath(opf, item.Href)
			if err != nil {
				return nil, fmt.Errorf("EPUB item %s: %w", id, err)
			}
			aliases = append(aliases, name)
			switch strings.ToLower(item.MIME) {
			case "application/xhtml+xml", "text/html", "image/svg+xml":
				doc = archiveDocument{Name: name, MIME: item.MIME, Aliases: aliases}
			}
			if doc.Name != "" {
				break
			}
			if item.Fallback == "" {
				return nil, fmt.Errorf("EPUB spine item %q (%s) has no supported XHTML/SVG fallback", id, item.MIME)
			}
			id = item.Fallback
		}
		if id != ref.ID {
			b.Warnings = append(b.Warnings, fmt.Sprintf("EPUB item %s uses fallback %s", ref.ID, id))
		}
		// Include non-linear spine entries too: endnotes and appendices are
		// part of a whole-book translation. Reused fallbacks are read once.
		if index, ok := seen[doc.Name]; ok {
			docs[index].Aliases = append(docs[index].Aliases, doc.Aliases...)
			continue
		}
		seen[doc.Name] = len(docs)
		docs = append(docs, doc)
	}
	if len(docs) == 0 {
		return nil, fmt.Errorf("EPUB has no readable spine")
	}
	b.Metadata["epub_documents"] = len(docs)
	return docs, nil
}

func epubEncryptedPaths(raw []byte) (map[string]bool, error) {
	var document struct {
		Entries []struct {
			Reference struct {
				URI string `xml:"URI,attr"`
			} `xml:"CipherData>CipherReference"`
		} `xml:"EncryptedData"`
	}
	if err := xml.Unmarshal(raw, &document); err != nil {
		return nil, fmt.Errorf("invalid EPUB encryption descriptor: %w", err)
	}
	paths := map[string]bool{}
	for _, entry := range document.Entries {
		name, err := archivePath("", entry.Reference.URI)
		if err != nil {
			return nil, err
		}
		paths[name] = true
	}
	return paths, nil
}
