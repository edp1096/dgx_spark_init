"""Add HTTP admission/idle reclamation to pinned qwentts; leave inference unchanged."""
import os
from pathlib import Path

root = Path(os.environ.get('QWENTTS_SOURCE_ROOT', '/src'))
p = root / 'src/tts-server.h'
s = p.read_text()

def replace(old, new):
    global s
    if s.count(old) != 1:
        raise RuntimeError('unexpected qwentts patch anchor: ' + old[:80])
    s = s.replace(old, new)

replace('#include <condition_variable>', '#include <condition_variable>\n#include <chrono>')
tracker = r'''
// Admission and quiesce share a lock. A PCM lease remains alive until its
// content provider is released and the inference thread has joined.
class TalkTtsLifecycle {
public:
    class Lease {
    public:
        explicit Lease(TalkTtsLifecycle& owner) : owner_(owner) {}
        ~Lease() { owner_.complete(active_); }
        void start() {
            std::lock_guard<std::mutex> lock(owner_.mutex_);
            active_ = true;
            ++owner_.active_;
        }
    private:
        TalkTtsLifecycle& owner_;
        bool active_ = false;
    };
    std::shared_ptr<Lease> admit() {
        std::lock_guard<std::mutex> lock(mutex_);
        // Keep HTTP workers available for health/quiesce while jobs wait.
        if (quiescing_ || jobs_ >= 4) return {};
        auto lease = std::make_shared<Lease>(*this);
        ++jobs_;
        return lease;
    }
    std::string snapshot(bool quiesce = false, bool resume = false) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (quiesce && jobs_ == 0) quiescing_ = true;
        if (resume) quiescing_ = false;
        const double idle = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - last_completed_).count();
        return "{\"status\":\"ok\",\"busy\":" + std::string(jobs_ ? "true" : "false")
            + ",\"active\":" + std::to_string(active_)
            + ",\"queued\":" + std::to_string(jobs_ - active_)
            + ",\"quiescing\":" + (quiescing_ ? "true" : "false")
            + ",\"idle_for_seconds\":" + std::to_string(idle)
            + ",\"memory_schema\":2,\"workspace_kind\":\"whole-service\","
              "\"memory_gib\":4.0,\"workspace_gib\":1.5,\"process_reclaim\":true}";
    }
    std::mutex synthesis_gate;
private:
    void complete(bool active) {
        std::lock_guard<std::mutex> lock(mutex_);
        --jobs_;
        if (active) --active_;
        last_completed_ = std::chrono::steady_clock::now();
    }
    std::mutex mutex_;
    size_t jobs_ = 0, active_ = 0;
    bool quiescing_ = false;
    std::chrono::steady_clock::time_point last_completed_ = std::chrono::steady_clock::now();
};
static TalkTtsLifecycle g_talk_tts;

'''
anchor = 'static void tts_handle_speech(const tts_backend & be, const httplib::Request & http_req, httplib::Response & res) {'
replace(anchor, tracker + anchor + r'''
    auto usage = g_talk_tts.admit();
    if (!usage) {
        const auto state = g_talk_tts.snapshot();
        res.status = state.find("\"quiescing\":true") != std::string::npos ? 503 : 429;
        res.set_content(state, "application/json");
        return;
    }
    res.set_header("X-Audio-Sample-Rate", "24000");
    res.set_header("X-Audio-Channels", "1");
    res.set_header("X-Audio-Sample-Format", "s16le");
''')
replace('''        int         rc = be.synthesize(req, sink, synth_err);''', '''        std::unique_lock<std::mutex> gate(g_talk_tts.synthesis_gate);
        usage->start();
        int         rc = be.synthesize(req, sink, synth_err);''')
replace('''    struct stream_state {
        std::mutex''', '''    struct stream_state {
        std::shared_ptr<TalkTtsLifecycle::Lease> usage;
        std::mutex''')
replace('''    auto st = std::make_shared<stream_state>();''', '''    auto st = std::make_shared<stream_state>();
    st->usage = usage;''')
replace('''    st->th = std::thread([&be, req, st]() {''', '''    st->th = std::thread([&be, req, st]() {
        std::unique_lock<std::mutex> gate(g_talk_tts.synthesis_gate);
        st->usage->start();''')
replace('''    httplib::Server svr;
    g_svr = &svr;''', '''    httplib::Server svr;
    svr.new_task_queue = [] { return new httplib::ThreadPool(8); };
    g_svr = &svr;''')
replace('''    svr.Get("/health", tts_handle_health);''', r'''
    svr.Get("/health", tts_handle_health);
    svr.Get("/ready", tts_handle_health);
    svr.Get("/v1/runtime/memory", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(g_talk_tts.snapshot(), "application/json");
    });
    svr.Post("/v1/runtime/quiesce", [](const httplib::Request&, httplib::Response& res) {
        const auto state = g_talk_tts.snapshot(true);
        res.status = state.find("\"busy\":true") != std::string::npos ? 409 : 200;
        res.set_content(state, "application/json");
    });
    svr.Post("/v1/runtime/resume", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(g_talk_tts.snapshot(false, true), "application/json");
    });
''')
p.write_text(s)
