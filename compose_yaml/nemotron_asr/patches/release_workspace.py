"""Patch pinned NeMo runtime to release workspaces without unloading weights."""
from pathlib import Path
import os

root = Path(os.environ.get('NEMO_SOURCE_ROOT', '/src'))
def replace(path, old, new):
    p = root / path
    text = p.read_text()
    if text.count(old) != 1:
        raise RuntimeError(f'{path}: expected one patch anchor, got {text.count(old)}')
    p.write_text(text.replace(old, new))

replace('src/runtime/ggml/runtime.h', '    int setup();',
        '    int setup();\n    void release_workspace();')
replace('src/runtime/ggml/session.cpp', 'Session::~Session() = default;', '''Session::~Session() = default;

// Keep model/state tensor containers and their weight buffers intact. Only
// request graphs, intermediates and scheduler activation pools are discarded.
void Session::release_workspace() {
    std::lock_guard<std::mutex> lock(backend_manager_->compute_mutex());
    for (auto backend : backends) ggml_backend_synchronize(backend);
    run_cache_.clear();
    run_cache_lru_.clear();
    sched.reset();
    std::vector<uint8_t>().swap(sched_meta);
    init_schedule();
    fprintf(stderr, "[sparktalk] model=%p weights=%zu workspace cleared",
            static_cast<void*>(model_tensor_container.get()),
            model_tensor_container->total_backend_buffer_bytes());
    fputc(10, stderr);
}
''')
replace('src/asr/recognizer.h', '    AsrModel* model() const { return model_.get(); }',
        '    void release_workspace();\n    AsrModel* model() const { return model_.get(); }')
replace('src/asr/recognizer.cpp', 'int\nRecognizer::sample_rate() const {', '''void
Recognizer::release_workspace() {
    if (active_offline_requests_.load() || active_streaming_ingress_.load()) return;
    for (const auto& entry : model_->diagnostic_sessions()) {
        if (entry.session) entry.session->release_workspace();
    }
    if (diar_model_) {
        if (diar_model_->model().session()) diar_model_->model().session()->release_workspace();
        if (diar_model_->fe().diagnostic_session()) diar_model_->fe().diagnostic_session()->release_workspace();
    }
}

int
Recognizer::sample_rate() const {''')
replace('src/asr/diar/diarizer.h', '    BatchMetrics batch_metrics() const;',
        '    BatchMetrics batch_metrics() const;\n    void release_workspace() const;')
replace('src/asr/diar/diarizer.cpp', 'int\nDiarizer::sample_rate() const {', '''void
Diarizer::release_workspace() const {
    if (model_->model().session()) model_->model().session()->release_workspace();
    if (model_->fe().diagnostic_session()) model_->fe().diagnostic_session()->release_workspace();
}

int
Diarizer::sample_rate() const {''')
# Explicitly opt in only for our serialized HTTP server deployment. The normal
# HTTP result, word timings, and speaker IDs are unchanged.
replace('server/http/http_server.cpp', '#include "http_server.h"',
        '#include "http_server.h"\n#include <cstdlib>\n#include <malloc.h>')
replace('server/http/http_server.cpp',
        '                const auto body = transcript_response(result, language, format);', '''                if (const auto* workspace_env = std::getenv("SPARKTALK_RELEASE_WORKSPACE"); workspace_env && std::string(workspace_env) == "1") {
                    recognizer->release_workspace();
                    malloc_trim(0);
                    fprintf(stderr, "[sparktalk] ASR workspace released; weights retained\\n");
                }
                const auto body = transcript_response(result, language, format);''')
replace('server/http/http_server.cpp',
        '                                  : asr::DiarizationMode::Streaming);', '''                                  : asr::DiarizationMode::Streaming);
            if (const auto* workspace_env = std::getenv("SPARKTALK_RELEASE_WORKSPACE"); workspace_env && std::string(workspace_env) == "1") {
                engine->release_workspace();
                malloc_trim(0);
                fprintf(stderr, "[sparktalk] diarization workspace released; weights retained\\n");
            }''')
