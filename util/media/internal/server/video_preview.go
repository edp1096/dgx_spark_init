package server

import (
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

const videoPreviewFilenameSuffix = ".timeline.jpg"

func (s *Server) videoPreviewPath(id string) string {
	return filepath.Join(s.dataDir, "previews", id+videoPreviewFilenameSuffix)
}

func (s *Server) deleteVideoPreview(id string) error {
	s.videoPreviewMu.Lock()
	defer s.videoPreviewMu.Unlock()
	for _, path := range []string{s.videoPreviewPath(id), s.videoPreviewPath(id) + ".tmp"} {
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	return nil
}

func (s *Server) ensureVideoPreview(id, source string) error {
	s.videoPreviewMu.Lock()
	defer s.videoPreviewMu.Unlock()

	destination := s.videoPreviewPath(id)
	if info, err := os.Stat(destination); err == nil && info.Size() > 0 {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
		return err
	}

	input, err := os.Open(source)
	if err != nil {
		return err
	}
	defer input.Close()

	reader, writer := io.Pipe()
	multipartWriter := multipart.NewWriter(writer)
	writeErr := make(chan error, 1)
	go func() {
		defer close(writeErr)
		part, err := multipartWriter.CreateFormFile("video", filepath.Base(source))
		if err == nil {
			_, err = io.Copy(part, input)
		}
		if closeErr := multipartWriter.Close(); err == nil {
			err = closeErr
		}
		_ = writer.CloseWithError(err)
		writeErr <- err
	}()

	endpoint := strings.TrimRight(s.config().Engines["media"].Endpoint, "/") + "/v1/media/thumbnails"
	request, err := http.NewRequest(http.MethodPost, endpoint, reader)
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", multipartWriter.FormDataContentType())
	response, err := s.client.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode/100 != 2 {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 1<<20))
		return fmt.Errorf("thumbnail engine returned %d: %s", response.StatusCode, strings.TrimSpace(string(body)))
	}

	temporary := destination + ".tmp"
	output, err := os.Create(temporary)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(output, io.LimitReader(response.Body, 16<<20))
	closeErr := output.Close()
	if copyErr != nil {
		_ = os.Remove(temporary)
		return copyErr
	}
	if closeErr != nil {
		_ = os.Remove(temporary)
		return closeErr
	}
	if err := <-writeErr; err != nil {
		_ = os.Remove(temporary)
		return err
	}
	return os.Rename(temporary, destination)
}

func (s *Server) videoJobPreview(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	job, ok := s.jobs.Get(id)
	if !ok || job.Kind != "video" || job.Status != "completed" || job.OutputURL == "" {
		http.NotFound(w, r)
		return
	}
	source := s.jobs.OutputPath(filepath.Base(job.OutputURL))
	if err := s.ensureVideoPreview(id, source); err != nil {
		http.Error(w, "video preview: "+err.Error(), http.StatusBadGateway)
		return
	}
	w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
	http.ServeFile(w, r, s.videoPreviewPath(id))
}
