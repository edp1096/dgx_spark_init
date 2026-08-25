package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

const assistantVisionImageLimit = 12

var assistantImageIndexPattern = regexp.MustCompile(`(?i)(?:#\s*)?(\d+)\s*번`)

type assistantRecentImage struct {
	Index  int    `json:"index"`
	JobID  string `json:"job_id"`
	Status string `json:"status"`
}

type assistantSheetImage struct {
	Index int
	Image image.Image
}

func (s *Server) assistantContactSheet(request assistantChatRequest) (string, []int, error) {
	if !assistantNeedsVision(request.Messages) {
		return "", nil, nil
	}
	recent, err := decodeAssistantRecentImages(request.State["recent_images"])
	if err != nil || len(recent) == 0 {
		return "", nil, err
	}
	mentioned := assistantMentionedImageIndices(latestAssistantUserMessage(request.Messages))
	tiles := make([]assistantSheetImage, 0, min(len(recent), assistantVisionImageLimit))
	indices := make([]int, 0, cap(tiles))
	for _, item := range recent {
		if len(tiles) >= assistantVisionImageLimit || item.Index < 1 || item.JobID == "" || item.Status != "completed" {
			continue
		}
		if len(mentioned) > 0 && !mentioned[item.Index] {
			continue
		}
		job, ok := s.jobs.Get(item.JobID)
		if !ok || job.Kind != "image" || job.Status != "completed" || job.OutputURL == "" {
			continue
		}
		file, openErr := os.Open(s.jobs.OutputPath(filepath.Base(job.OutputURL)))
		if openErr != nil {
			continue
		}
		decoded, _, decodeErr := image.Decode(io.LimitReader(file, 64<<20))
		_ = file.Close()
		if decodeErr != nil {
			continue
		}
		tiles = append(tiles, assistantSheetImage{Index: item.Index, Image: decoded})
		indices = append(indices, item.Index)
	}
	if len(tiles) == 0 {
		return "", nil, fmt.Errorf("no decodable recent images")
	}
	sheet := renderAssistantContactSheet(tiles)
	var encoded bytes.Buffer
	if err := jpeg.Encode(&encoded, sheet, &jpeg.Options{Quality: 88}); err != nil {
		return "", nil, err
	}
	return "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(encoded.Bytes()), indices, nil
}

func decodeAssistantRecentImages(value any) ([]assistantRecentImage, error) {
	if value == nil {
		return nil, nil
	}
	data, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	var images []assistantRecentImage
	if err := json.Unmarshal(data, &images); err != nil {
		return nil, err
	}
	return images, nil
}

func assistantNeedsVision(messages []assistantChatMessage) bool {
	latest := strings.ToLower(latestAssistantUserMessage(messages))
	if latest == "" {
		return false
	}
	cues := []string{
		"보이", "찾아", "골라", "어느 이미지", "어떤 이미지", "무슨 이미지", "어느 사진", "어떤 사진",
		"이미지 내용", "사진 내용", "직접 보고", "실제로", "설명해", "묘사해", "비교해", "비교하",
		"더 정면", "더 나은", "무슨 색", "어떤 색", "어떤 옷", "무슨 옷", "자세가", "구도가", "방향이",
	}
	for _, cue := range cues {
		if strings.Contains(latest, cue) {
			return true
		}
	}
	return false
}

func latestAssistantUserMessage(messages []assistantChatMessage) string {
	for index := len(messages) - 1; index >= 0; index-- {
		if strings.EqualFold(strings.TrimSpace(messages[index].Role), "user") {
			return strings.TrimSpace(messages[index].Content)
		}
	}
	return ""
}

func assistantMentionedImageIndices(content string) map[int]bool {
	indices := map[int]bool{}
	for _, match := range assistantImageIndexPattern.FindAllStringSubmatch(content, -1) {
		if len(match) != 2 {
			continue
		}
		if value, err := strconv.Atoi(match[1]); err == nil && value > 0 {
			indices[value] = true
		}
	}
	return indices
}

func renderAssistantContactSheet(tiles []assistantSheetImage) *image.RGBA {
	cellWidth, cellHeight, columns := 320, 240, min(3, len(tiles))
	if len(tiles) == 1 {
		cellWidth, cellHeight, columns = 768, 768, 1
	} else if len(tiles) == 2 {
		cellWidth, cellHeight, columns = 512, 512, 2
	}
	rows := (len(tiles) + columns - 1) / columns
	destination := image.NewRGBA(image.Rect(0, 0, columns*cellWidth, rows*cellHeight))
	draw.Draw(destination, destination.Bounds(), &image.Uniform{C: color.RGBA{R: 13, G: 17, B: 21, A: 255}}, image.Point{}, draw.Src)
	for position, tile := range tiles {
		x := position % columns * cellWidth
		y := position / columns * cellHeight
		cell := image.Rect(x+3, y+3, x+cellWidth-3, y+cellHeight-3)
		drawScaledContain(destination, cell, tile.Image)
		drawSheetLabel(destination, x+10, y+10, "#"+strconv.Itoa(tile.Index))
	}
	return destination
}

func drawScaledContain(destination *image.RGBA, target image.Rectangle, source image.Image) {
	bounds := source.Bounds()
	if bounds.Dx() < 1 || bounds.Dy() < 1 {
		return
	}
	scale := min(float64(target.Dx())/float64(bounds.Dx()), float64(target.Dy())/float64(bounds.Dy()))
	width := max(1, int(float64(bounds.Dx())*scale))
	height := max(1, int(float64(bounds.Dy())*scale))
	left := target.Min.X + (target.Dx()-width)/2
	top := target.Min.Y + (target.Dy()-height)/2
	for y := 0; y < height; y++ {
		sourceY := bounds.Min.Y + y*bounds.Dy()/height
		for x := 0; x < width; x++ {
			sourceX := bounds.Min.X + x*bounds.Dx()/width
			destination.Set(left+x, top+y, source.At(sourceX, sourceY))
		}
	}
}

var sheetGlyphs = map[rune][]string{
	'#': {"01010", "11111", "01010", "01010", "11111", "01010", "01010"},
	'0': {"01110", "10001", "10011", "10101", "11001", "10001", "01110"},
	'1': {"00100", "01100", "00100", "00100", "00100", "00100", "01110"},
	'2': {"01110", "10001", "00001", "00010", "00100", "01000", "11111"},
	'3': {"11110", "00001", "00001", "01110", "00001", "00001", "11110"},
	'4': {"00010", "00110", "01010", "10010", "11111", "00010", "00010"},
	'5': {"11111", "10000", "10000", "11110", "00001", "00001", "11110"},
	'6': {"01110", "10000", "10000", "11110", "10001", "10001", "01110"},
	'7': {"11111", "00001", "00010", "00100", "01000", "01000", "01000"},
	'8': {"01110", "10001", "10001", "01110", "10001", "10001", "01110"},
	'9': {"01110", "10001", "10001", "01111", "00001", "00001", "01110"},
}

func drawSheetLabel(destination *image.RGBA, left, top int, label string) {
	const pixel, gap = 4, 3
	width := len([]rune(label))*5*pixel + max(0, len([]rune(label))-1)*gap
	background := image.Rect(left-6, top-6, left+width+6, top+7*pixel+6)
	draw.Draw(destination, background, &image.Uniform{C: color.RGBA{R: 4, G: 7, B: 9, A: 255}}, image.Point{}, draw.Src)
	x := left
	for _, character := range label {
		glyph := sheetGlyphs[character]
		for row, bits := range glyph {
			for column, bit := range bits {
				if bit != '1' {
					continue
				}
				rectangle := image.Rect(x+column*pixel, top+row*pixel, x+(column+1)*pixel, top+(row+1)*pixel)
				draw.Draw(destination, rectangle, &image.Uniform{C: color.RGBA{R: 255, G: 192, B: 47, A: 255}}, image.Point{}, draw.Src)
			}
		}
		x += 5*pixel + gap
	}
}
