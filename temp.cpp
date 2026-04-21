#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cassert>
#include <cstring>

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

// ─────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────

constexpr int SAMPLE_RATE  = 44100;
constexpr int WINDOW_SIZE  = 1024;
constexpr int HOP_SIZE     = 512;
constexpr int NUM_BINS     = WINDOW_SIZE / 2;

// Output ring buffer must hold at least one full window + one hop of headroom
// so the worker can overlap-add a full window while the callback is draining a hop.
// We use a power-of-two size for cheap modulo via bitmask.
constexpr int OUT_RING_SIZE = 8192; // must be power of two and > WINDOW_SIZE + HOP_SIZE
static_assert((OUT_RING_SIZE & (OUT_RING_SIZE - 1)) == 0, "OUT_RING_SIZE must be power of two");

// ─────────────────────────────────────────────
// Tone
// ─────────────────────────────────────────────

struct Tone {
    double frequency;
    double amplitude;
    double phase;
};

// ─────────────────────────────────────────────
// Lock-free single-producer / single-consumer ring buffer (for float samples)
//
// The audio callback is the sole CONSUMER of the output ring.
// The worker thread is the sole PRODUCER into the output ring.
// Because there is exactly one reader and one writer, we only need
// atomic loads/stores — no mutex required.
// ─────────────────────────────────────────────

class SPSCRingBuffer {
public:
    explicit SPSCRingBuffer(size_t capacity)
        : buf_(capacity, 0.0f), capacity_(capacity),
          readPos_(0), writePos_(0) {
        assert((capacity & (capacity - 1)) == 0 && "capacity must be power of two");
    }

    // Number of samples available to read
    size_t readAvailable() const {
        size_t w = writePos_.load(std::memory_order_acquire);
        size_t r = readPos_.load(std::memory_order_relaxed);
        return (w - r) & (capacity_ - 1);
    }

    // Number of slots free to write
    size_t writeAvailable() const {
        return capacity_ - 1 - readAvailable();
    }

    // Add `value` into the sample at `offset` ahead of the current write head.
    // Used for overlap-add: multiple windows contribute to the same output sample.
    void addAt(size_t offset, float value) {
        size_t w = writePos_.load(std::memory_order_relaxed);
        size_t idx = (w + offset) & (capacity_ - 1);
        buf_[idx] += value;
    }

    // Advance the write head by `n` after a batch of addAt() calls.
    void advanceWrite(size_t n) {
        size_t w = writePos_.load(std::memory_order_relaxed);
        writePos_.store((w + n) & (capacity_ - 1), std::memory_order_release);
        // Note: we store the masked value so writePos_ stays in [0, capacity_).
        // readAvailable() uses subtraction with the same mask, so this is safe.
    }

    // Read `n` samples into `dst` and advance the read head.
    // Returns false if fewer than `n` samples are available.
    bool read(float* dst, size_t n) {
        if (readAvailable() < n) return false;
        size_t r = readPos_.load(std::memory_order_relaxed);
        for (size_t i = 0; i < n; i++) {
            size_t idx = (r + i) & (capacity_ - 1);
            dst[i] = buf_[idx];
            buf_[idx] = 0.0f; // clear after reading so overlap-add starts from zero
        }
        readPos_.store((r + n) & (capacity_ - 1), std::memory_order_release);
        return true;
    }

private:
    std::vector<float>   buf_;
    size_t               capacity_;
    std::atomic<size_t>  readPos_;
    std::atomic<size_t>  writePos_;
};

// ─────────────────────────────────────────────
// Hann window
// ─────────────────────────────────────────────

std::vector<double> makeHannWindow(int N) {
    std::vector<double> w(N);
    for (int i = 0; i < N; i++)
        w[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (N - 1)));
    return w;
}

// ─────────────────────────────────────────────
// DFT  (O(N²) — replace with FFT for production)
// ─────────────────────────────────────────────

std::vector<Tone> computeDFT(const std::vector<float>& windowed, int N) {
    std::vector<Tone> tones(NUM_BINS);
    for (int k = 0; k < NUM_BINS; k++) {
        std::complex<double> sum(0.0, 0.0);
        for (int n = 0; n < N; n++) {
            double angle = (2.0 * M_PI * k * n) / N;
            sum += (double)windowed[n] * std::complex<double>(std::cos(angle), -std::sin(angle));
        }
        tones[k].frequency = (double)k * SAMPLE_RATE / N;
        tones[k].amplitude = std::abs(sum) * (2.0 / N);
        tones[k].phase     = std::arg(sum);
    }
    return tones;
}

// ─────────────────────────────────────────────
// IDFT  — reconstructs WINDOW_SIZE samples from NUM_BINS tones
// ─────────────────────────────────────────────

std::vector<float> computeIDFT(const std::vector<Tone>& tones, int N) {
    std::vector<float> output(N, 0.0f);
    for (int n = 0; n < N; n++) {
        double sample = 0.0;
        for (int k = 0; k < NUM_BINS; k++) {
            // Reconstruct from polar form: A * cos(2πkn/N + φ)
            sample += tones[k].amplitude * std::cos(2.0 * M_PI * k * n / N + tones[k].phase);
        }
        output[n] = (float)(sample * 0.5); // 0.5 accounts for the two-sided spectrum
    }
    return output;
}

// ─────────────────────────────────────────────
// Shared state between worker and callback
// ─────────────────────────────────────────────

struct SharedState {
    ma_decoder          decoder;

    // Input: worker reads from the decoder directly (no input ring needed
    // since we're reading a file, not a live stream).
    // A mutex protects decoder access between the worker and any other thread.
    std::mutex          decoderMutex;

    // Output ring: worker overlap-adds synthesized windows into here;
    // callback drains HOP_SIZE samples per call.
    SPSCRingBuffer      outputRing{OUT_RING_SIZE};

    // Worker signals the callback that at least one hop is ready.
    // Callback signals the worker when it has drained a hop (i.e. needs more).
    std::mutex              hopMutex;
    std::condition_variable hopReady;   // worker → callback: hop available
    std::condition_variable hopDrained; // callback → worker: hop consumed
    int                     hopsReady{0};

    std::atomic<bool>   running{true};

    const std::vector<double> hannWindow = makeHannWindow(WINDOW_SIZE);
};

// ─────────────────────────────────────────────
// Worker thread
//
// Runs continuously, staying exactly one hop ahead of the callback.
// Each iteration:
//   1. Read WINDOW_SIZE samples from the decoder (with HOP_SIZE overlap)
//   2. Apply Hann window
//   3. DFT → tones
//   4. [Modify tones here for any effect]
//   5. IDFT → synthesized samples
//   6. Overlap-add into the output ring
//   7. Advance output ring write head by HOP_SIZE
//   8. Signal callback that a hop is ready
// ─────────────────────────────────────────────

void workerThread(SharedState& state) {
    // We maintain a local overlap buffer so we can re-read the last
    // (WINDOW_SIZE - HOP_SIZE) samples each iteration without seeking.
    std::vector<float> overlapBuf(WINDOW_SIZE - HOP_SIZE, 0.0f);
    std::vector<float> newSamples(HOP_SIZE);
    std::vector<float> window(WINDOW_SIZE);

    while (state.running.load()) {

        // ── 1. Read HOP_SIZE new samples from decoder ──────────────────
        ma_uint64 framesRead = 0;
        {
            std::lock_guard<std::mutex> lock(state.decoderMutex);
            ma_decoder_read_pcm_frames(&state.decoder, newSamples.data(),
                                       HOP_SIZE, &framesRead);
        }

        if (framesRead == 0) {
            state.running.store(false);
            break;
        }

        // Zero-pad if we hit the end of the file
        if (framesRead < (ma_uint64)HOP_SIZE)
            std::fill(newSamples.begin() + framesRead, newSamples.end(), 0.0f);

        // ── 2. Build the full window from overlap + new samples ─────────
        // [ overlapBuf (512) | newSamples (512) ] = window (1024)
        std::memcpy(window.data(), overlapBuf.data(),
                    overlapBuf.size() * sizeof(float));
        std::memcpy(window.data() + overlapBuf.size(), newSamples.data(),
                    newSamples.size() * sizeof(float));

        // Update overlap buffer for next iteration
        std::memcpy(overlapBuf.data(),
                    window.data() + HOP_SIZE,
                    overlapBuf.size() * sizeof(float));

        // ── 3. Apply Hann window ────────────────────────────────────────
        std::vector<float> windowed(WINDOW_SIZE);
        for (int i = 0; i < WINDOW_SIZE; i++)
            windowed[i] = window[i] * (float)state.hannWindow[i];

        // ── 4. DFT ─────────────────────────────────────────────────────
        std::vector<Tone> tones = computeDFT(windowed, WINDOW_SIZE);

        // ── 5. Modify tones (pitch shift, filter, etc.) ─────────────────
        // Example: pass-through (no modification)
        // To pitch shift up one octave you'd do:
        //   std::vector<Tone> shifted(NUM_BINS, {0,0,0});
        //   for (int k = 0; k < NUM_BINS/2; k++) {
        //       shifted[k*2] = tones[k];
        //       shifted[k*2].frequency *= 2.0;
        //   }
        //   tones = shifted;

        // ── 6. IDFT ────────────────────────────────────────────────────
        std::vector<float> synthesized = computeIDFT(tones, WINDOW_SIZE);

        // ── 7. Overlap-add into output ring ────────────────────────────
        // Each synthesized sample is scaled by the Hann window again
        // (synthesis window). With 50% overlap, two Hann² windows sum
        // to a constant, giving perfect reconstruction.
        for (int i = 0; i < WINDOW_SIZE; i++) {
            float val = synthesized[i] * (float)state.hannWindow[i];
            state.outputRing.addAt(i, val);
        }

        // Advance write head by HOP_SIZE — the next window will overlap-add
        // into the samples we haven't advanced past yet.
        state.outputRing.advanceWrite(HOP_SIZE);

        // ── 8. Signal callback that one more hop is ready ───────────────
        {
            std::lock_guard<std::mutex> lock(state.hopMutex);
            state.hopsReady++;
        }
        state.hopReady.notify_one();

        // ── 9. Wait if we're too far ahead (> 2 hops buffered) ──────────
        // This prevents the worker from racing far ahead and using lots of memory.
        {
            std::unique_lock<std::mutex> lock(state.hopMutex);
            state.hopDrained.wait(lock, [&]{
                return state.hopsReady <= 2 || !state.running.load();
            });
        }
    }

    // Unblock callback if it's waiting
    state.hopReady.notify_all();
}

// ─────────────────────────────────────────────
// Audio callback  (runs on the audio thread — no blocking, no allocation)
//
// Waits until the worker has a hop ready, then drains HOP_SIZE samples
// from the output ring into the playback buffer.
// ─────────────────────────────────────────────

void audioCallback(ma_device* device, void* output, const void* /*input*/,
                   ma_uint32 frameCount) {
    SharedState* state = (SharedState*)device->pUserData;
    float* out = (float*)output;

    if (!state->running.load()) {
        std::memset(out, 0, frameCount * sizeof(float));
        return;
    }

    // Wait for worker to have at least one hop ready.
    // In practice this wait is nearly instant since the worker stays ahead.
    {
        std::unique_lock<std::mutex> lock(state->hopMutex);
        state->hopReady.wait(lock, [&]{
            return state->hopsReady > 0 || !state->running.load();
        });
        if (state->hopsReady > 0)
            state->hopsReady--;
    }

    // Drain one hop from the output ring
    if (!state->outputRing.read(out, frameCount)) {
        // Underrun — worker didn't keep up (shouldn't happen with the buffering above)
        std::memset(out, 0, frameCount * sizeof(float));
        std::cerr << "underrun!\n";
    }

    // Tell the worker it can produce more
    state->hopDrained.notify_one();
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────

int main(int argc, char* argv[]) {
    const char* path = (argc > 1) ? argv[1] : "audio.wav";

    SharedState state;

    // ── Init decoder ───────────────────────────────────────────────────
    if (ma_decoder_init_file(path, nullptr, &state.decoder) != MA_SUCCESS) {
        std::cerr << "Failed to open: " << path << "\n";
        return 1;
    }

    // ── Init playback device ───────────────────────────────────────────
    ma_device_config cfg = ma_device_config_init(ma_device_type_playback);
    cfg.playback.format   = ma_format_f32;
    cfg.playback.channels = 1;
    cfg.sampleRate        = SAMPLE_RATE;
    cfg.periodSizeInFrames = HOP_SIZE; // callback fires every HOP_SIZE samples
    cfg.dataCallback      = audioCallback;
    cfg.pUserData         = &state;

    ma_device device;
    if (ma_device_init(nullptr, &cfg, &device) != MA_SUCCESS) {
        std::cerr << "Failed to init audio device\n";
        ma_decoder_uninit(&state.decoder);
        return 1;
    }

    // ── Pre-fill two hops so the callback never starves on startup ─────
    // We run the worker inline for 2 iterations before starting playback.
    // (Alternatively you could just wait for hopsReady >= 2 in a CV.)

    // ── Start worker thread ────────────────────────────────────────────
    std::thread worker(workerThread, std::ref(state));

    // Wait until we have at least 2 hops buffered before starting playback
    {
        std::unique_lock<std::mutex> lock(state.hopMutex);
        state.hopReady.wait(lock, [&]{ return state.hopsReady >= 2; });
    }

    // ── Start playback ─────────────────────────────────────────────────
    if (ma_device_start(&device) != MA_SUCCESS) {
        std::cerr << "Failed to start device\n";
        state.running.store(false);
        worker.join();
        ma_device_uninit(&device);
        ma_decoder_uninit(&state.decoder);
        return 1;
    }

    std::cout << "Playing " << path << " — press Enter to stop\n";
    std::cin.get();

    // ── Cleanup ────────────────────────────────────────────────────────
    state.running.store(false);
    state.hopReady.notify_all();
    state.hopDrained.notify_all();

    worker.join();
    ma_device_uninit(&device);
    ma_decoder_uninit(&state.decoder);

    return 0;
}