package server

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"hash/crc32"
	"time"
	"unicode/utf16"

	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
)

var pngSignature = []byte{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a}

type imageEXIFMetadata struct {
	Version         int            `json:"version"`
	Software        string         `json:"software"`
	Creator         string         `json:"creator,omitempty"`
	Copyright       string         `json:"copyright,omitempty"`
	Website         string         `json:"website,omitempty"`
	Note            string         `json:"note,omitempty"`
	JobID           string         `json:"job_id"`
	CreatedAt       time.Time      `json:"created_at"`
	Model           string         `json:"model,omitempty"`
	Mode            string         `json:"mode,omitempty"`
	Prompt          string         `json:"prompt"`
	EffectivePrompt string         `json:"effective_prompt,omitempty"`
	Width           int            `json:"width,omitempty"`
	Height          int            `json:"height,omitempty"`
	Seed            any            `json:"seed,omitempty"`
	Parameters      map[string]any `json:"parameters,omitempty"`
}

func metadataForImageJob(j jobs.Job, effectivePrompt string, profile config.ImageMetadata) imageEXIFMetadata {
	metadata := imageEXIFMetadata{
		Version: 1, Software: "SparkMediaPanel", JobID: j.ID, CreatedAt: j.CreatedAt,
		Creator: profile.Creator, Copyright: profile.Copyright, Website: profile.Website, Note: profile.Note,
		Prompt: j.Prompt, EffectivePrompt: effectivePrompt, Parameters: j.Params,
	}
	if effectivePrompt == j.Prompt {
		metadata.EffectivePrompt = ""
	}
	if value, ok := j.Params["model"].(string); ok {
		metadata.Model = value
	}
	if value, ok := j.Params["mode"].(string); ok {
		metadata.Mode = value
	}
	metadata.Width = intFromMetadata(j.Params["width"])
	metadata.Height = intFromMetadata(j.Params["height"])
	metadata.Seed = j.Params["seed"]
	return metadata
}

func intFromMetadata(value any) int {
	switch number := value.(type) {
	case int:
		return number
	case int64:
		return int(number)
	case float64:
		return int(number)
	default:
		return 0
	}
}

func embedImageEXIF(data []byte, metadata imageEXIFMetadata) []byte {
	if len(data) < 33 || !bytes.Equal(data[:8], pngSignature) || string(data[12:16]) != "IHDR" {
		return data
	}
	payload, err := json.Marshal(metadata)
	if err != nil {
		return data
	}
	exif := buildTIFFEXIF(metadata, payload)
	chunk := make([]byte, 12+len(exif))
	binary.BigEndian.PutUint32(chunk[:4], uint32(len(exif)))
	copy(chunk[4:8], "eXIf")
	copy(chunk[8:8+len(exif)], exif)
	binary.BigEndian.PutUint32(chunk[8+len(exif):], crc32.ChecksumIEEE(chunk[4:8+len(exif)]))

	insertAt := 8 + 12 + int(binary.BigEndian.Uint32(data[8:12]))
	result := make([]byte, 0, len(data)+len(chunk))
	result = append(result, data[:insertAt]...)
	result = append(result, chunk...)
	result = append(result, data[insertAt:]...)
	return result
}

func buildTIFFEXIF(metadata imageEXIFMetadata, payload []byte) []byte {
	strings := []struct {
		tag  uint16
		data []byte
	}{
		{0x010e, append([]byte(metadata.Prompt), 0)},
		{0x0131, append([]byte(metadata.Software), 0)},
		{0x0132, append([]byte(metadata.CreatedAt.Local().Format("2006:01:02 15:04:05")), 0)},
	}
	if metadata.Creator != "" {
		strings = append(strings, struct {
			tag  uint16
			data []byte
		}{0x013b, append([]byte(metadata.Creator), 0)})
	}
	if metadata.Copyright != "" {
		strings = append(strings, struct {
			tag  uint16
			data []byte
		}{0x8298, append([]byte(metadata.Copyright), 0)})
	}
	commentBytes := encodeUnicodeUserComment(payload)

	ifd0Entries := len(strings) + 1
	ifd0Size := 2 + ifd0Entries*12 + 4
	result := make([]byte, 8+ifd0Size)
	copy(result[:2], "II")
	binary.LittleEndian.PutUint16(result[2:4], 42)
	binary.LittleEndian.PutUint32(result[4:8], 8)
	binary.LittleEndian.PutUint16(result[8:10], uint16(ifd0Entries))
	for index, value := range strings {
		entry := result[10+index*12 : 22+index*12]
		binary.LittleEndian.PutUint16(entry[0:2], value.tag)
		binary.LittleEndian.PutUint16(entry[2:4], 2)
		binary.LittleEndian.PutUint32(entry[4:8], uint32(len(value.data)))
		if len(value.data) <= 4 {
			copy(entry[8:12], value.data)
		} else {
			binary.LittleEndian.PutUint32(entry[8:12], uint32(len(result)))
			result = append(result, value.data...)
		}
	}
	if len(result)%2 != 0 {
		result = append(result, 0)
	}
	exifIFDOffset := len(result)
	const exifIFDSize = 2 + 12 + 4
	commentOffset := exifIFDOffset + exifIFDSize
	pointerEntry := result[10+len(strings)*12 : 22+len(strings)*12]
	writeTIFFEntry(pointerEntry, 0x8769, 4, 1, uint32(exifIFDOffset))
	result = append(result, make([]byte, exifIFDSize+len(commentBytes))...)
	binary.LittleEndian.PutUint16(result[exifIFDOffset:exifIFDOffset+2], 1)
	writeTIFFEntry(result[exifIFDOffset+2:exifIFDOffset+14], 0x9286, 7, uint32(len(commentBytes)), uint32(commentOffset))
	copy(result[commentOffset:], commentBytes)
	return result
}

func writeTIFFEntry(target []byte, tag, kind uint16, count, value uint32) {
	binary.LittleEndian.PutUint16(target[0:2], tag)
	binary.LittleEndian.PutUint16(target[2:4], kind)
	binary.LittleEndian.PutUint32(target[4:8], count)
	binary.LittleEndian.PutUint32(target[8:12], value)
}

func encodeUnicodeUserComment(payload []byte) []byte {
	units := utf16.Encode([]rune(string(payload)))
	result := make([]byte, 8+len(units)*2)
	copy(result[:8], []byte{'U', 'N', 'I', 'C', 'O', 'D', 'E', 0})
	for index, unit := range units {
		binary.BigEndian.PutUint16(result[8+index*2:], unit)
	}
	return result
}

func extractImageEXIF(data []byte) (imageEXIFMetadata, bool) {
	if len(data) < 8 || !bytes.Equal(data[:8], pngSignature) {
		return imageEXIFMetadata{}, false
	}
	for offset := 8; offset+12 <= len(data); {
		length := int(binary.BigEndian.Uint32(data[offset : offset+4]))
		end := offset + 12 + length
		if length < 0 || end > len(data) {
			return imageEXIFMetadata{}, false
		}
		if string(data[offset+4:offset+8]) == "eXIf" {
			return extractTIFFMetadata(data[offset+8 : offset+8+length])
		}
		offset = end
	}
	return imageEXIFMetadata{}, false
}

func extractTIFFMetadata(data []byte) (imageEXIFMetadata, bool) {
	if len(data) < 14 || string(data[:2]) != "II" || binary.LittleEndian.Uint16(data[2:4]) != 42 {
		return imageEXIFMetadata{}, false
	}
	ifdOffset := int(binary.LittleEndian.Uint32(data[4:8]))
	if ifdOffset+2 > len(data) {
		return imageEXIFMetadata{}, false
	}
	count := int(binary.LittleEndian.Uint16(data[ifdOffset : ifdOffset+2]))
	exifOffset := 0
	for index := 0; index < count; index++ {
		entry := ifdOffset + 2 + index*12
		if entry+12 > len(data) {
			return imageEXIFMetadata{}, false
		}
		if binary.LittleEndian.Uint16(data[entry:entry+2]) == 0x8769 {
			exifOffset = int(binary.LittleEndian.Uint32(data[entry+8 : entry+12]))
			break
		}
	}
	if exifOffset <= 0 || exifOffset+2 > len(data) {
		return imageEXIFMetadata{}, false
	}
	exifCount := int(binary.LittleEndian.Uint16(data[exifOffset : exifOffset+2]))
	for index := 0; index < exifCount; index++ {
		entry := exifOffset + 2 + index*12
		if entry+12 > len(data) || binary.LittleEndian.Uint16(data[entry:entry+2]) != 0x9286 {
			continue
		}
		length := int(binary.LittleEndian.Uint32(data[entry+4 : entry+8]))
		offset := int(binary.LittleEndian.Uint32(data[entry+8 : entry+12]))
		if length < 8 || offset < 0 || offset+length > len(data) {
			return imageEXIFMetadata{}, false
		}
		payload := decodeUnicodeUserComment(data[offset : offset+length])
		var metadata imageEXIFMetadata
		if len(payload) == 0 || json.Unmarshal(payload, &metadata) != nil {
			return imageEXIFMetadata{}, false
		}
		return metadata, true
	}
	return imageEXIFMetadata{}, false
}

func decodeUnicodeUserComment(data []byte) []byte {
	if len(data) < 8 || string(data[:8]) != "UNICODE\x00" || (len(data)-8)%2 != 0 {
		return nil
	}
	units := make([]uint16, (len(data)-8)/2)
	for index := range units {
		units[index] = binary.BigEndian.Uint16(data[8+index*2:])
	}
	return []byte(string(utf16.Decode(units)))
}
