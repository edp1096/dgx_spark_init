package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

type engineHTTPError struct {
	StatusCode int
	Body       string
}

func (e *engineHTTPError) Error() string {
	return fmt.Sprintf("engine returned %d: %s", e.StatusCode, e.Body)
}

func (s *Server) callJSON(url string, payload any) ([]byte, string, error) {
	return s.callJSONContext(context.Background(), url, payload)
}

func (s *Server) callJSONContext(ctx context.Context, url string, payload any) ([]byte, string, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, "", err
	}
	req.Header.Set("Content-Type", "application/json")
	return s.do(req)
}

func (s *Server) callMultipart(url string, fields map[string]string, fileField string, paths []string) ([]byte, string, error) {
	return s.callMultipartContext(context.Background(), url, fields, fileField, paths)
}

func (s *Server) callMultipartContext(ctx context.Context, url string, fields map[string]string, fileField string, paths []string) ([]byte, string, error) {
	return s.callMultipartFilesContext(ctx, url, fields, map[string][]string{fileField: paths})
}

func (s *Server) callMultipartFilesContext(ctx context.Context, url string, fields map[string]string, files map[string][]string) ([]byte, string, error) {
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	for k, v := range fields {
		_ = mw.WriteField(k, v)
	}
	for fileField, paths := range files {
		for _, p := range paths {
			f, e := os.Open(p)
			if e != nil {
				return nil, "", e
			}
			part, e := mw.CreateFormFile(fileField, filepath.Base(p))
			if e == nil {
				_, e = io.Copy(part, f)
			}
			f.Close()
			if e != nil {
				return nil, "", e
			}
		}
	}
	_ = mw.Close()
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, url, &body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	return s.do(req)
}

func (s *Server) callMultipartToFile(url string, fields map[string]string, fileField string, paths []string, output string) (http.Header, error) {
	return s.callMultipartFilesToFile(url, fields, map[string][]string{fileField: paths}, output)
}

func (s *Server) callMultipartFilesToFile(url string, fields map[string]string, files map[string][]string, output string) (http.Header, error) {
	return s.callMultipartFilesToFileContext(context.Background(), url, fields, files, output)
}

func (s *Server) callMultipartFilesToFileContext(ctx context.Context, url string, fields map[string]string, files map[string][]string, output string) (http.Header, error) {
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	for k, v := range fields {
		_ = mw.WriteField(k, v)
	}
	for field, paths := range files {
		for _, p := range paths {
			f, err := os.Open(p)
			if err != nil {
				return nil, err
			}
			part, err := mw.CreateFormFile(field, filepath.Base(p))
			if err == nil {
				_, err = io.Copy(part, f)
			}
			_ = f.Close()
			if err != nil {
				return nil, err
			}
		}
	}
	if err := mw.Close(); err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, &body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return nil, &engineHTTPError{StatusCode: resp.StatusCode, Body: strings.TrimSpace(string(data))}
	}
	dst, err := os.Create(output)
	if err != nil {
		return nil, err
	}
	_, copyErr := io.Copy(dst, resp.Body)
	closeErr := dst.Close()
	if copyErr != nil {
		return nil, copyErr
	}
	return resp.Header.Clone(), closeErr
}

func (s *Server) do(req *http.Request) ([]byte, string, error) {
	resp, e := s.client.Do(req)
	if e != nil {
		return nil, "", e
	}
	defer resp.Body.Close()
	data, e := io.ReadAll(io.LimitReader(resp.Body, 100<<20))
	if e != nil {
		return nil, "", e
	}
	if resp.StatusCode/100 != 2 {
		return nil, "", &engineHTTPError{StatusCode: resp.StatusCode, Body: strings.TrimSpace(string(data))}
	}
	return data, resp.Header.Get("Content-Type"), nil
}
