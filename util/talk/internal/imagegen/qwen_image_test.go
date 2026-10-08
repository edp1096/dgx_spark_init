package imagegen

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestNativeCapabilitiesReplaceLegacyCatalog(t *testing.T) {
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"model":"qwen-image-2.1-uc-nvfp4","operations":["generate","background_remove"],"styles":[],"data":[]}`))
	}))
	defer api.Close()
	result, err := New(api.URL, "test", 0).Capabilities(context.Background())
	if err != nil || result.Model != "qwen-image-2.1-uc-nvfp4" || len(result.Styles) != 0 || len(result.Operations) != 2 {
		t.Fatalf("wrong native capabilities %+v %v", result, err)
	}
}
