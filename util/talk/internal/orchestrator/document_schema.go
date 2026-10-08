package orchestrator

import "encoding/json"

// DocumentToolSchema uses the same versioned schema as the embedded renderer.
func DocumentToolSchema() json.RawMessage {
	data, err := assets.ReadFile("assets/extra-documents/document-schema.json")
	if err != nil {
		panic(err)
	}
	return json.RawMessage(data)
}
