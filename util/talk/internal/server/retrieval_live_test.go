package server

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/embedding"
	"testing"
	"time"
)

func TestRetrievalLiveKoreanGPU(t *testing.T) {
	endpoint := os.Getenv("SPARKTALK_EMBEDDING_LIVE_ENDPOINT")
	if endpoint == "" {
		t.Skip("requires prepared CUDA embedding worker")
	}
	d, err := db.Open(filepath.Join(t.TempDir(), "quality.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	records := [][3]string{
		{"QAD 운영", "QAD는 KV 캐시 512K와 MTP 3으로 설정한다. 언어 모델 본체와 이미지 DiT, ASR, TTS를 메모리에 상주시킨다.", "언어 모델을 계속 올려두고 그림과 음성까지 같이 쓰려면 문맥 크기를 얼마로 줄였지?"},
		{"NVFP4 구성", "RadixArk NVFP4 세트는 1M KV 문맥으로 언어 모델과 음성 전사를 사용한다. 이미지 생성과 답변 음성은 포함하지 않는다.", "백만 토큰을 쓰는 radixark 세트에서 영상 속 말을 글로 바꾸는 것은 되나?"},
		{"EXL3 설정", "EXL3는 1M 문맥, Q8 KV, MTP 3을 사용한다. Qwen Image와 MiniMax H3의 DiT를 함께 상주시킨다.", "exl3에서 이미지와 동영상 생성 본체를 같이 올려두는 구성은?"},
		{"음성 속도", "Qwen3-TTS 답변 읽기는 기본 속도 1.0이다. Speak rate를 1.2 또는 1.3으로 올리면 음높이를 유지하며 빠르게 들을 수 있다.", "대답을 읽어주는 목소리가 답답하게 느린데 더 빨리 듣고 싶어."},
		{"이미지 메모리", "이미지 생성은 DiT 본체를 유지하고 텍스트 인코더 TE와 VAE를 작업 단계마다 적재하고 해제한다.", "그림 그리는 본체는 그대로 두면서 잠깐 쓰는 인코더와 디코더만 반환하는 방법은?"},
		{"이미지 게시 승인", "새로운 GHCR 이미지 저장소에 게시할 때는 항목과 이름을 사용자에게 먼저 승인받는다. 기존 이미지의 갱신은 승인된 범위에서 진행한다.", "새 컨테이너를 레지스트리에 올리려면 먼저 누구에게 무엇을 확인해야 하지?"},
		{"검색 구조", "SQLite FTS5는 바이너리에 내장된 키워드 검색이다. EmbeddingGemma의 의미 검색과 독립적으로 후보를 찾아 RRF로 순위를 합친다.", "단어가 일치하는 방식과 뜻이 비슷한 자료를 찾는 방식을 어떻게 같이 쓰지?"},
		{"ASR 모델 공개", "Nemotron ASR Q5_K 모델은 edp1096/nemotron-3.5-asr-streaming-0.6b-gguf에 공개했다. 화자 구분은 NVIDIA 공식 Q8 GGUF를 사용한다.", "우리 음성 인식 양자화 파일을 공개한 허깅페이스 주소는?"},
		{"README 정리", "SparkTalk README는 실행, 모델 준비, 설정, 개발 빌드에 필요한 내용만 짧게 남긴다. 내부 실험 기록을 쌓아두지 않는다.", "사용 설명서가 너무 장황해지지 않게 무엇만 남기기로 했지?"},
		{"SSH 인증", "Extra SSH는 개인키 인증을 사용한다. 개인키 파일은 권한 0600으로 보관하고 알려진 서버의 호스트 지문을 확인한다.", "원격 명령을 실행할 때 어떤 로그인 방식으로 인증하나?"},
		{"문서 생성", "Extra Documents는 DOCX, PPTX, XLSX와 PDF를 만든다. 원본 문서와 PDF를 대화 첨부에서 내려받는다.", "워드 문서와 발표 슬라이드를 만들어 첨부하는 서비스는 뭐지?"},
		{"TTS 가중치", "Qwen3-TTS 0.6B는 CustomVoice 본체 Q8_0와 오디오 codec Q8_0 GGUF를 사용한다. Sohee가 기본 화자다.", "답변 음성의 본체와 코덱은 각각 어느 정밀도로 쓰고 화자는 누구야?"},
		{"한국어 혼합 발음", "Magpie TTS는 한국어 문장 안의 영어 이름과 약어를 제대로 읽지 못하는 문제가 있어 Qwen3-TTS로 교체했다.", "한글 문장에 섞인 영문 약자를 씹던 음성 모델을 왜 바꿨지?"},
		{"기동 여유", "서비스 작업을 시작하기 전에 시스템 가용 메모리와 CUDA 즉시 여유를 확인한다. 최소 시스템 여유 1.5 GiB를 보호한다.", "새 작업을 시작하기 전에 실제 남은 램을 확인하는 기준은?"},
		{"첨부 전사", "영상과 오디오 파일은 Extra Media에서 16kHz 모노 PCM으로 변환한 뒤 Nemotron ASR로 전사한다.", "첨부한 녹음 파일의 말을 텍스트로 바꾸기 전에 어떤 변환을 하나?"},
		{"영상 생성", "MiniMax H3는 음성을 포함하는 864×480, 24fps, 약 5.17초의 MP4를 생성한다.", "우리가 만드는 짧은 동영상의 해상도와 초당 프레임 수는?"},
		{"코드 버전", "대화의 코드 프로젝트는 SQLite에 버전을 보관한다. 이전 버전 복원은 그 내용을 새로운 버전으로 저장한다.", "웹 코드를 잘못 고쳤는데 전에 저장한 상태로 되돌릴 수 있나?"},
		{"스캔 자료", "텍스트가 없는 스캔 PDF와 이미지는 OCR 실행으로 글자를 읽고 문서 지식의 검색 색인을 갱신한다.", "종이 책을 찍은 PDF에서 글씨를 검색하려면 먼저 뭘 해야 해?"},
		{"임베딩 정밀도", "EmbeddingGemma 2는 BF16 또는 FP32로 실행한다. FP16은 NaN이 발생할 수 있다. 출력 기본 차원은 768이다.", "의미 검색 모델이 숫자 오류를 내지 않게 어떤 실수형으로 돌리지?"},
		{"지원 서비스", "Extra Media, Collector, Documents는 요청 시 실행한다. 언어 모델과 상주 모델의 프로세스를 불필요하게 종료하지 않는다.", "보조 도구를 쓰려고 매번 언어 모델 전체를 내렸다 올리는 건 아니지?"},
	}
	ids := make([]string, len(records))
	for i, row := range records {
		ids[i] = fmt.Sprintf("%032x", i+1)
		doc, _, e := d.AddKnowledgeDocument(db.KnowledgeDocument{ID: ids[i], CollectionID: 1, Title: row[0], SHA256: fmt.Sprintf("hash-%d", i), StoragePath: fmt.Sprintf("objects/%d", i), Status: "processing"})
		if e != nil {
			t.Fatal(e)
		}
		if e = d.ReplaceKnowledgeChunks(doc.ID, []db.KnowledgeChunk{{Ordinal: 0, PageStart: i + 1, PageEnd: i + 1, Content: row[1]}}, len(records), "ready", ""); e != nil {
			t.Fatal(e)
		}
	}
	sources, err := d.PendingEmbeddings(100)
	if err != nil {
		t.Fatal(err)
	}
	for _, source := range sources {
		response, e := embedding.Encode(context.Background(), endpoint, "document", []embedding.Input{{Title: source.Title, Text: source.Content}})
		if e != nil {
			t.Fatal(e)
		}
		var parts []db.EmbeddingSegment
		for _, p := range response.Data {
			parts = append(parts, db.EmbeddingSegment{Text: p.Text, Vector: p.Vector})
		}
		if e = d.CompleteEmbedding(source, embedding.Profile, parts); e != nil {
			t.Fatal(e)
		}
	}
	s := &Server{db: d, cfg: config.Config{Runtime: config.RuntimeConfig{Mode: "external"}, Embedding: config.EmbeddingConfig{Enabled: true, Endpoint: endpoint, Timeout: "30s", MinSimilarity: .62}}}
	result := map[string]any{}
	details := []map[string]any{}
	lexicalTop5, semanticTop5, hybridTop5 := 0, 0, 0
	for i, row := range records {
		started := time.Now()
		lexical, e := d.SearchKnowledge(row[2], 1, 5)
		if e != nil {
			t.Fatal(e)
		}
		vector, e := s.queryVector(context.Background(), row[2])
		if e != nil {
			t.Fatal(e)
		}
		semantic, e := d.SearchSemantic(context.Background(), embedding.Profile, "knowledge", "", 0, 1, vector, 0, 5)
		if e != nil {
			t.Fatal(e)
		}
		hybrid, e := s.hybridKnowledge(context.Background(), row[2], 1, 5)
		if e != nil {
			t.Fatal(e)
		}
		lr, sr, hr := 0, 0, 0
		score := 0.0
		for j, x := range lexical {
			if x.DocumentID == ids[i] {
				lr = j + 1
			}
		}
		for j, x := range semantic {
			if x.Source.DocumentID == ids[i] {
				sr = j + 1
				score = x.Similarity
			}
		}
		for j, x := range hybrid {
			if x.DocumentID == ids[i] {
				hr = j + 1
			}
		}
		if lr > 0 {
			lexicalTop5++
		}
		if sr > 0 {
			semanticTop5++
		}
		if hr > 0 {
			hybridTop5++
		}
		details = append(details, map[string]any{"question": row[2], "expected": row[0], "fts_rank": lr, "semantic_rank": sr, "hybrid_rank": hr, "similarity": score, "elapsed_ms": time.Since(started).Milliseconds()})
	}
	negatives := []map[string]any{}
	for _, query := range []string{"오늘 부산의 날씨와 우산이 필요한지 알려줘", "사과 파이를 오븐에 굽는 온도와 시간", "이탈리아 로마 여행의 호텔을 추천해 줘", "강아지가 먹으면 안 되는 음식은?", "조선 시대 세종의 업적을 설명해 줘"} {
		v, e := s.queryVector(context.Background(), query)
		if e != nil {
			t.Fatal(e)
		}
		matches, e := d.SearchSemantic(context.Background(), embedding.Profile, "knowledge", "", 0, 1, v, 0, 1)
		if e != nil {
			t.Fatal(e)
		}
		filtered, e := s.hybridKnowledge(context.Background(), query, 1, 5)
		if e != nil {
			t.Fatal(e)
		}
		if len(filtered) != 0 {
			t.Fatalf("unrelated query recalled documents: %s %+v", query, filtered)
		}
		negatives = append(negatives, map[string]any{"question": query, "top_similarity": matches[0].Similarity, "top_title": matches[0].Source.Title})
	}
	result["queries"] = len(records)
	result["fts_top5"] = lexicalTop5
	result["semantic_top5"] = semanticTop5
	result["hybrid_top5"] = hybridTop5
	result["details"] = details
	result["unrelated_questions"] = negatives
	b, _ := json.MarshalIndent(result, "", "  ")
	if path := os.Getenv("SPARKTALK_RETRIEVAL_REPORT"); path != "" {
		if err = os.WriteFile(path, b, 0600); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("Korean fixture top5: FTS=%d/%d semantic=%d/%d hybrid=%d/%d", lexicalTop5, len(records), semanticTop5, len(records), hybridTop5, len(records))
	if hybridTop5 < lexicalTop5 {
		t.Fatal("hybrid retrieval regressed on Korean fixture")
	}
}
