package orchestrator

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"encoding/json"
	"io"
	"io/fs"
	"os"
	"path"
	"strings"
	"testing"
)

func TestEveryFirstInstallBuildInputIsEmbedded(t *testing.T) {
	cat, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	manifests, err := os.ReadFile("recipe_sources/packages.json")
	if err != nil {
		t.Fatal(err)
	}
	var packages []struct {
		ID string `json:"id"`
	}
	if err = json.Unmarshal(manifests, &packages); err != nil {
		t.Fatal(err)
	}
	covered := map[string]bool{}
	for _, p := range packages {
		covered[p.ID] = true
	}
	for _, c := range cat.Components {
		t.Run(c.ID, func(t *testing.T) {
			files := map[string][]byte{}
			if id := recipeID(c); id != "" {
				if !covered[id] {
					t.Fatalf("recipe %s missing from packaging check", id)
				}
				raw, e := assets.ReadFile("assets/recipes/" + id + ".tar.gz")
				if e != nil {
					t.Fatal(e)
				}
				gz, e := gzip.NewReader(bytes.NewReader(raw))
				if e != nil {
					t.Fatal(e)
				}
				defer gz.Close()
				tr := tar.NewReader(gz)
				for {
					h, e := tr.Next()
					if e == io.EOF {
						break
					}
					if e != nil {
						t.Fatal(e)
					}
					if h.Typeflag == tar.TypeReg {
						files[h.Name], e = io.ReadAll(tr)
						if e != nil {
							t.Fatal(e)
						}
					}
				}
				if len(files["runtime.sh"]) == 0 {
					t.Fatal("missing lifecycle")
				}
			} else {
				if _, e := composeAsset(c.ComposeAsset); e != nil {
					t.Fatal(e)
				}
				build := embeddedBuildAsset(c.ComposeAsset)
				if build == "" {
					t.Fatal("no first-install build for registered service")
				}
				root := "assets/" + build
				e := fs.WalkDir(assets, root, func(p string, d fs.DirEntry, e error) error {
					if e != nil {
						return e
					}
					if !d.IsDir() {
						b, e := assets.ReadFile(p)
						if e != nil {
							return e
						}
						files[strings.TrimSuffix(strings.TrimPrefix(p, root+"/"), ".asset")] = b
					}
					return nil
				})
				if e != nil {
					t.Fatal(e)
				}
			}
			for name, data := range files {
				// DSpark pulls a pinned published image; upstream overlay Dockerfiles are not build inputs of its install path.
				if recipeID(c) == "ds4fve" && strings.HasPrefix(name, "upstream/") {
					continue
				}
				if !strings.HasPrefix(path.Base(name), "Dockerfile") {
					continue
				}
				root := path.Dir(name)
				for _, line := range strings.Split(strings.ReplaceAll(string(data), "\\\n", " "), "\n") {
					fields := strings.Fields(strings.TrimSpace(line))
					if len(fields) == 0 {
						continue
					}
					if fields[0] == "FROM" && len(fields) > 1 && strings.HasPrefix(fields[1], "dgx-") {
						t.Errorf("%s requires unavailable local base %s", name, fields[1])
					}
					if fields[0] != "COPY" || strings.Contains(line, "--from=") {
						continue
					}
					for _, src := range fields[1 : len(fields)-1] {
						if strings.HasPrefix(src, "--") {
							continue
						}
						src = path.Clean(path.Join(root, src))
						found := false
						for file := range files {
							match, _ := path.Match(src, file)
							if src == "." || match || strings.HasPrefix(file, src+"/") {
								found = true
								break
							}
						}
						if !found {
							t.Errorf("%s COPY input missing: %s", name, src)
						}
					}
				}
			}
		})
	}
	if _, err := assets.ReadFile("assets/flux2-paint/phased/__init__.py"); err != nil {
		t.Fatal("FLUX python module omitted", err)
	}
}

func TestPublishedQADDoesNotRequireLocalConversionRecords(t *testing.T) {
	c := Component{ComposeAsset: "compose.flash-next.yaml", RuntimeOptions: map[string]string{"MODEL_VARIANT": "huihui_lil"}}
	items := componentModelAssets(c)
	if len(items) != 1 || items[0].Repo != QwenQADHuihuiLIL || items[0].Revision == "local" || items[0].Path != "/hf/"+QwenQADHuihuiLIL {
		t.Fatalf("invalid published preparation: %+v", items)
	}
}
