"""Extend the pinned native TTS HTTP runtime; no inference wrappers or weight changes."""
from pathlib import Path
import os
root=Path(os.environ.get('NEMO_SOURCE_ROOT','/src'))
p=root/'server/http/http_server.cpp'
s=p.read_text()
def replace(old,new):
 global s
 if s.count(old)!=1:raise RuntimeError('unexpected native TTS patch anchor: '+old[:80])
 s=s.replace(old,new)
replace('#include <condition_variable>','#include <condition_variable>\n#include <chrono>')
anchor='struct Server::Impl {'
tracker=r'''// Protect accepted TTS inference, including requests waiting for the native gate.
// A guard survives parsing and synthesis exceptions; quiesce and admission share
// the same lock so process reclamation cannot race a newly admitted GPU job.
class TtsRuntimeLifecycle {
 public:
    class Lease {
     public:
        explicit Lease(TtsRuntimeLifecycle& owner) : owner_(owner) {}
        ~Lease() { owner_.complete(); }
     private:
        TtsRuntimeLifecycle& owner_;
    };
    std::unique_ptr<Lease> admit() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (quiescing_) return {};
        auto lease = std::make_unique<Lease>(*this);
        ++jobs_;
        return lease;
    }
    Value snapshot(bool ready, bool quiesce = false, bool resume = false) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (quiesce && jobs_ == 0) quiescing_ = true;
        if (resume) quiescing_ = false;
        Value body(Value::Object{});
        body["status"] = ready ? "ok" : "not_ready";
        body["busy"] = jobs_ != 0;
        body["active"] = static_cast<double>(jobs_ != 0);
        body["queued"] = static_cast<double>(jobs_ > 0 ? jobs_ - 1 : 0);
        body["quiescing"] = quiescing_;
        body["idle_for_seconds"] = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - last_completed_).count();
        body["memory_schema"] = 2.0;
        body["workspace_kind"] = "whole-service";
        body["memory_gib"] = 3.0;
        body["workspace_gib"] = 1.5;
        body["process_reclaim"] = true;
        return body;
    }
 private:
    void complete() {
        std::lock_guard<std::mutex> lock(mutex_);
        --jobs_;
        last_completed_ = std::chrono::steady_clock::now();
    }
    std::mutex mutex_;
    size_t jobs_ = 0;
    bool quiescing_ = false;
    std::chrono::steady_clock::time_point last_completed_ = std::chrono::steady_clock::now();
};

'''
replace(anchor,tracker+anchor)
replace('    TtsPreemptionCoordinator tts_preemption;','    TtsPreemptionCoordinator tts_preemption;\n    TtsRuntimeLifecycle tts_runtime;\n    std::mutex tts_request_gate;')
replace('        server->Get("/version",',r'''        server->Get("/v1/runtime/memory", [this](const httplib::Request&, httplib::Response& response) {
            const bool ready = this->models.ready();
            response.status = ready ? 200 : 503;
            response.set_content(this->tts_runtime.snapshot(ready).dump(), "application/json");
        });
        server->Post("/v1/runtime/quiesce", [this](const httplib::Request&, httplib::Response& response) {
            const bool ready = this->models.ready();
            auto state = this->tts_runtime.snapshot(ready, true);
            response.status = !ready ? 503 : state.bool_or("busy", true) ? 409 : 200;
            response.set_content(state.dump(), "application/json");
        });
        server->Post("/v1/runtime/resume", [this](const httplib::Request&, httplib::Response& response) {
            response.set_content(this->tts_runtime.snapshot(this->models.ready(), false, true).dump(), "application/json");
        });
        server->Get("/version",''')
replace('''#if defined(NEMO_SPEECH_REGISTRY_TTS)
                const Value body = Value::parse(request.body);''',r'''#if defined(NEMO_SPEECH_REGISTRY_TTS)
                auto usage = this->tts_runtime.admit();
                if (!usage) {
                    fail(response, 503, "TTS is returning idle memory; restart the runtime before retrying");
                    return;
                }
                std::unique_lock<std::mutex> request_gate(this->tts_request_gate, std::defer_lock);
                if (!this->config.preempt_tts) request_gate.lock();
                const Value body = Value::parse(request.body);''')
p.write_text(s)
