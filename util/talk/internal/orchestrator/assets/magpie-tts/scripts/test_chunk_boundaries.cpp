// Compile the actual NeMo-Speech.cpp alignment code; no models or GPU required.
#include <algorithm>
#include <cstdio>
#include <vector>
#include "tts/magpietts/model.h"

namespace tts = nemo_speech::tts;

int main() {
    tts::magpietts_hparams h;
    h.apply_attention_prior = true;
    h.n_cross_head = 1;
    h.n_cross_dhead = 1;
    h.attention_prior_lookahead_window = 6;
    h.attention_prior_advance_threshold = 8;
    auto require = [](bool ok, const char* message) {
        if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
    };
    // A short heading after a long sentence must not expose the old sentence.
    for (int current : {1, 4, 15, 40}) {
        tts::MagpieLongformAttentionPriorState state;
        state.beginChunk(h, 0, 30, 30, true);
        const int history = 10, left = 20, length = history + current;
        state.beginChunk(h, left, length, current, false);
        require(state.lastAttendedAbsolute() >= 30, "new chunk rewound into spoken text");
        for (int step = 0; step < 20; ++step) {
            const auto& prior = state.prior();
            require(prior.size() == size_t(length), "invalid prior shape");
            require(std::all_of(prior.begin(), prior.begin()+history,
                               [](float v) { return v == 0; }), "history became audible");
            require(std::any_of(prior.begin()+history, prior.end(),
                               [](float v) { return v > 0; }), "current text was masked");
            std::vector<float> scores(length, 0.0f);
            scores[history-1] = 100.0f; // Deliberately distract with old text.
            scores.back() = 1.0f;
            state.update(h, step, length, scores);
            require(state.lastAttendedAbsolute() >= 30, "alignment selected old text");
            require(state.lastAttendedRelative() < length, "alignment exceeded window");
        }
    }
    // Real attention at the last phonemes must advance immediately, not wait
    // eight decoder steps per excluded token and repeat the final word.
    tts::MagpieLongformAttentionPriorState tail;
    tail.beginChunk(h, 0, 8, 8, true);
    std::vector<float> scores(8, 0.0f);
    scores[4] = 1.0f;
    tail.update(h, 0, 8, scores);
    require(tail.lastAttendedRelative() == 4, "failed to reach tail");
    for (int pos : {5, 6, 7}) {
        std::fill(scores.begin(), scores.end(), 0.0f);
        scores[pos] = 1.0f;
        tail.update(h, pos, 8, scores);
        require(tail.lastAttendedRelative() == pos, "last phoneme excluded from alignment");
    }
    require(tts::magpietts_frames_to_emit(2, 2, 1) == 1, "frame before EOS lost");
    require(tts::magpietts_frames_to_emit(2, 2, 0) == 0, "EOS frame emitted");
    require(tts::magpietts_frames_to_emit(2, 2, -1) == 2, "valid stacked frame lost");
    std::puts("chunk history, final-phoneme alignment and EOS frame tests passed");
}
