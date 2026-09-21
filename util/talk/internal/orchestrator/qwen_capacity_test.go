package orchestrator

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestQADCapacityRejectsSilentKVTruncation(t *testing.T) {
	for _, capacity := range []int{844992, 1048576} {
		t.Run(fmt.Sprint(capacity), func(t *testing.T) {
			api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/server_info" {
					t.Error(r.URL.Path)
				}
				fmt.Fprintf(w, `{"context_length":1048576,"max_total_num_tokens":%d}`, capacity)
			}))
			defer api.Close()
			c, _ := NewController()
			component := Component{ComposeAsset: "compose.flash-next.yaml", HealthURL: api.URL + "/health"}
			err := c.checkQwenCapacity(context.Background(), component)
			if capacity < 1048576 {
				if err == nil || !strings.Contains(err.Error(), "844992") {
					t.Fatal(err)
				}
			} else if err != nil {
				t.Fatal(err)
			}
		})
	}
}
