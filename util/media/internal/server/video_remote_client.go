package server

import (
	"context"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"sort"
	"strings"
)

func (s *Server) callRemoteVideoMultipartToFile(ctx context.Context, endpoint, sourceURL string, fields map[string]string, output string) (http.Header, error) {
	sourceRequest, err := http.NewRequestWithContext(ctx, http.MethodGet, sourceURL, nil)
	if err != nil {
		return nil, err
	}
	sourceResponse, err := s.client.Do(sourceRequest)
	if err != nil {
		return nil, err
	}
	defer sourceResponse.Body.Close()
	if sourceResponse.StatusCode/100 != 2 {
		data, _ := io.ReadAll(io.LimitReader(sourceResponse.Body, 1<<20))
		return nil, fmt.Errorf("source media returned %d: %s", sourceResponse.StatusCode, strings.TrimSpace(string(data)))
	}

	reader, writer := io.Pipe()
	multipartWriter := multipart.NewWriter(writer)
	producerDone := make(chan error, 1)
	go func() {
		var produceErr error
		defer func() {
			if produceErr == nil {
				produceErr = multipartWriter.Close()
			}
			_ = writer.CloseWithError(produceErr)
			producerDone <- produceErr
		}()
		keys := make([]string, 0, len(fields))
		for key := range fields {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			if produceErr = multipartWriter.WriteField(key, fields[key]); produceErr != nil {
				return
			}
		}
		part, partErr := multipartWriter.CreateFormFile("video", "source.mp4")
		if partErr != nil {
			produceErr = partErr
			return
		}
		_, produceErr = io.Copy(part, sourceResponse.Body)
	}()
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, reader)
	if err != nil {
		_ = reader.CloseWithError(err)
		return nil, err
	}
	request.Header.Set("Content-Type", multipartWriter.FormDataContentType())
	response, err := s.client.Do(request)
	if err != nil {
		_ = reader.CloseWithError(err)
		<-producerDone
		return nil, err
	}
	defer response.Body.Close()
	if producerErr := <-producerDone; producerErr != nil {
		return nil, producerErr
	}
	if response.StatusCode/100 != 2 {
		data, _ := io.ReadAll(io.LimitReader(response.Body, 4<<20))
		return nil, fmt.Errorf("engine returned %d: %s", response.StatusCode, strings.TrimSpace(string(data)))
	}
	destination, err := os.Create(output)
	if err != nil {
		return nil, err
	}
	_, copyErr := io.Copy(destination, response.Body)
	closeErr := destination.Close()
	if copyErr != nil {
		return nil, copyErr
	}
	return response.Header.Clone(), closeErr
}
