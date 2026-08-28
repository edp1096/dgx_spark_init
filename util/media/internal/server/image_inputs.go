package server

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	"image/png"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

type imageJobInputInfo struct {
	Role string `json:"role"`
	Name string `json:"name"`
	URL  string `json:"url"`
	Ref  string `json:"ref"`
}

func (s *Server) imageJobInputs(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	job, ok := s.jobs.Get(id)
	if !ok || job.Kind != "image" {
		http.NotFound(w, r)
		return
	}
	inputs := make([]imageJobInputInfo, 0, 8)
	for _, role := range []string{"reference", "identity", "sequence_character", "identity_reference", "identity_mask", "strict_mask", "depth", "vision", "style_reference", "nk2e", "anypaint", "anypaint_mask", "garment_source", "garment_reference", "face_swap_target", "face_swap_source"} {
		files, err := s.imageInputFiles(id, role)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		for index, path := range files {
			name := filepath.Base(path)
			switch role {
			case "identity":
				name = "identity" + filepath.Ext(path)
			case "sequence_character":
				name = "sequence-character" + filepath.Ext(path)
			case "identity_reference":
				name = fmt.Sprintf("identity-reference-%d%s", index+1, filepath.Ext(path))
			case "identity_mask":
				name = "identity-focus-mask" + filepath.Ext(path)
			case "strict_mask":
				name = "strict-change-mask" + filepath.Ext(path)
			case "depth":
				name = "depth" + filepath.Ext(path)
			case "vision":
				name = fmt.Sprintf("vision-%d%s", index+1, filepath.Ext(path))
			case "style_reference":
				name = fmt.Sprintf("style-reference-%d%s", index+1, filepath.Ext(path))
			case "nk2e":
				name = "nk2e" + filepath.Ext(path)
			case "anypaint":
				name = "anypaint-source" + filepath.Ext(path)
			case "anypaint_mask":
				name = "anypaint-mask" + filepath.Ext(path)
			case "garment_source":
				name = "garment-source" + filepath.Ext(path)
			case "garment_reference":
				name = fmt.Sprintf("garment-reference-%d%s", index+1, filepath.Ext(path))
			case "face_swap_target":
				name = "face-swap-target" + filepath.Ext(path)
			case "face_swap_source":
				name = "face-swap-source" + filepath.Ext(path)
			}
			inputs = append(inputs, imageJobInputInfo{
				Role: role,
				Name: name,
				URL:  fmt.Sprintf("/api/jobs/%s/inputs/%s/%d", id, role, index),
				Ref:  fmt.Sprintf("%s:%s:%d", id, role, index),
			})
		}
	}
	writeJSON(w, http.StatusOK, inputs)
}

func (s *Server) imageJobInput(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	job, ok := s.jobs.Get(id)
	if !ok || job.Kind != "image" {
		http.NotFound(w, r)
		return
	}
	index, err := strconv.Atoi(r.PathValue("index"))
	if err != nil || index < 0 {
		http.NotFound(w, r)
		return
	}
	files, err := s.imageInputFiles(id, r.PathValue("role"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if index >= len(files) {
		http.NotFound(w, r)
		return
	}
	http.ServeFile(w, r, files[index])
}

func (s *Server) imageInputFiles(id, role string) ([]string, error) {
	root := filepath.Join(s.dataDir, "inputs", id)
	dir := root
	switch role {
	case "output":
		job, ok := s.jobs.Get(id)
		if !ok || job.Kind != "image" || job.Status != "completed" || job.OutputURL == "" {
			return nil, nil
		}
		path := s.jobs.OutputPath(filepath.Base(job.OutputURL))
		if _, err := os.Stat(path); err != nil {
			if errors.Is(err, os.ErrNotExist) {
				return nil, nil
			}
			return nil, err
		}
		return []string{path}, nil
	case "reference":
	case "identity":
		dir = filepath.Join(root, "identity")
	case "sequence_character":
		dir = filepath.Join(root, "sequence-character")
	case "identity_reference":
		dir = filepath.Join(root, "identity-reference")
	case "identity_mask":
		dir = filepath.Join(root, "identity-mask")
	case "strict_mask":
		dir = filepath.Join(root, "strict-mask")
	case "depth":
		dir = filepath.Join(root, "depth")
	case "vision":
		dir = filepath.Join(root, "vision")
	case "style_reference":
		dir = filepath.Join(root, "style-reference")
	case "nk2e":
		dir = filepath.Join(root, "nk2e")
	case "anypaint":
		dir = filepath.Join(root, "anypaint")
	case "anypaint_mask":
		dir = filepath.Join(root, "anypaint-mask")
	case "garment_source":
		dir = filepath.Join(root, "garment-source")
	case "garment_reference":
		dir = filepath.Join(root, "garment-reference")
	case "face_swap_target":
		dir = filepath.Join(root, "face-swap-target")
	case "face_swap_source":
		dir = filepath.Join(root, "face-swap-source")
	case "sequence-previous":
		dir = filepath.Join(root, "sequence-previous")
	case "sequence-master":
		dir = filepath.Join(root, "sequence-master")
	case "sequence-draft":
		dir = filepath.Join(root, "sequence-draft")
	default:
		return nil, nil
	}
	entries, err := os.ReadDir(dir)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	files := make([]string, 0, len(entries))
	for _, entry := range entries {
		if entry.Type().IsRegular() {
			files = append(files, filepath.Join(dir, entry.Name()))
		}
	}
	sort.Strings(files)
	return files, nil
}

func (s *Server) appendReusedImageInputs(r *http.Request, field, dir string, max int, paths []string) ([]string, error) {
	tokens := r.MultipartForm.Value[field]
	if len(paths)+len(tokens) > max {
		return nil, fmt.Errorf("too many files for %s (maximum %d)", field, max)
	}
	if len(tokens) == 0 {
		return paths, nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	result := append([]string(nil), paths...)
	for _, token := range tokens {
		parts := strings.Split(token, ":")
		if len(parts) != 3 || parts[0] == "" || parts[1] == "" {
			return nil, fmt.Errorf("invalid stored image reference")
		}
		job, ok := s.jobs.Get(parts[0])
		if !ok || job.Kind != "image" {
			return nil, fmt.Errorf("stored image reference no longer exists")
		}
		index, err := strconv.Atoi(parts[2])
		if err != nil || index < 0 {
			return nil, fmt.Errorf("invalid stored image reference")
		}
		files, err := s.imageInputFiles(parts[0], parts[1])
		if err != nil {
			return nil, err
		}
		if index >= len(files) {
			return nil, fmt.Errorf("stored image reference no longer exists")
		}
		source := files[index]
		destination := filepath.Join(dir, fmt.Sprintf("%d%s", len(result), strings.ToLower(filepath.Ext(source))))
		if err := linkOrCopyFile(source, destination); err != nil {
			return nil, err
		}
		result = append(result, destination)
	}
	return result, nil
}

func linkOrCopyFile(source, destination string) error {
	if err := os.Link(source, destination); err == nil {
		return nil
	}
	input, err := os.Open(source)
	if err != nil {
		return err
	}
	defer input.Close()
	output, err := os.OpenFile(destination, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(output, input)
	closeErr := output.Close()
	if copyErr != nil {
		_ = os.Remove(destination)
		return copyErr
	}
	if closeErr != nil {
		_ = os.Remove(destination)
		return closeErr
	}
	return nil
}

func saveUploads(r *http.Request, field, dir string, max int) ([]string, error) {
	files := r.MultipartForm.File[field]
	if len(files) > max {
		return nil, fmt.Errorf("too many files (max %d)", max)
	}
	if len(files) == 0 {
		return nil, nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	out := make([]string, 0, len(files))
	for i, h := range files {
		src, e := h.Open()
		if e != nil {
			return nil, e
		}
		// Reference pixels are conditioning data, not merely previews. Saving a
		// PNG upload or a CDN response back as lossy WebP changed alpha edges and
		// was enough to make Krea Identity Edit retain the original clothing.
		// Decode every supported upload and persist one lossless PNG representation
		// so direct uploads, URL images and later job retries use identical pixels.
		data, readErr := io.ReadAll(io.LimitReader(src, (32<<20)+1))
		src.Close()
		if readErr != nil {
			return nil, readErr
		}
		if len(data) == 0 || len(data) > 32<<20 {
			return nil, fmt.Errorf("image upload must be between 1 byte and 32 MiB")
		}
		decoded, _, decodeErr := image.Decode(bytes.NewReader(data))
		if decodeErr != nil {
			// Preserve the old opaque-upload behavior for non-image engine test
			// fixtures and forward-compatible formats that this Go build cannot
			// decode. Recognized images always take the lossless PNG path below.
			name := fmt.Sprintf("%d%s", i, strings.ToLower(filepath.Ext(h.Filename)))
			dstPath := filepath.Join(dir, name)
			if writeErr := os.WriteFile(dstPath, data, 0o644); writeErr != nil {
				return nil, writeErr
			}
			out = append(out, dstPath)
			continue
		}
		name := fmt.Sprintf("%d.png", i)
		dstPath := filepath.Join(dir, name)
		dst, e := os.OpenFile(dstPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
		if e != nil {
			return nil, e
		}
		encodeErr := png.Encode(dst, decoded)
		closeErr := dst.Close()
		if encodeErr != nil {
			_ = os.Remove(dstPath)
			return nil, encodeErr
		}
		if closeErr != nil {
			_ = os.Remove(dstPath)
			return nil, closeErr
		}
		out = append(out, dstPath)
	}
	return out, nil
}
