#include <iostream>
#include <vector>
#include <complex>
#include <thread>
#include <atomic>
#include <cmath>
#include <chrono>
#include <algorithm>

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "raylib.h"

constexpr const char* AUDIO_PATH = "zaudio/ketamina.wav";

constexpr int SAMPLE_RATE = 44100;
constexpr int CHANNELS = 1;
constexpr ma_format FORMAT = ma_format_f32;

constexpr int WINDOW_SIZE = 1 << 10; // 2^10=1024
constexpr int HOP_SIZE = WINDOW_SIZE / 2;

constexpr int RING_BUFFER_SIZE = std::max(WINDOW_SIZE * 3, 1024);

constexpr int NUM_TONES = std::min(HOP_SIZE, 10);

std::atomic<bool> keepRunning{true};

ma_pcm_rb g_rb; // global ring buffer

struct Tone{
    float frequency;
    float amplitude;
    float phase;
};

std::vector<Tone> g_visualBuffers[2]; 
std::atomic<int> g_latestBufferIdx{0};

void loadWav(const char* filename, std::vector<float>& data) {
    ma_decoder decoder;
    ma_decoder_config config = ma_decoder_config_init(FORMAT, CHANNELS, SAMPLE_RATE);

    if (ma_decoder_init_file(filename, &config, &decoder) != MA_SUCCESS) {
        std::cerr << "failed to open file: " << filename << std::endl;
        return;
    }

    ma_uint64 frameCount;
    ma_decoder_get_length_in_pcm_frames(&decoder, &frameCount);

    data.resize(frameCount * CHANNELS);

    ma_uint64 framesRead;
    ma_decoder_read_pcm_frames(&decoder, data.data(), frameCount, &framesRead);
    ma_decoder_uninit(&decoder);
}

void fft(std::vector<std::complex<float>>& x) {
    size_t n = x.size();
    
    // bit reversal pre shuffle
    for (size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(x[i], x[j]);
    }

    // butterfly computations
    for (size_t len = 2; len <= n; len <<= 1) {
        float ang = 2 * M_PI / len;
        std::complex<float> wlen(std::cos(ang), -std::sin(ang)); // twiddle factor
        for (size_t i = 0; i < n; i += len) {
            std::complex<float> w(1);
            for (size_t j = 0; j < len / 2; j++) {
                std::complex<float> u = x[i + j], v = x[i + j + len / 2] * w;
                x[i + j] = u + v;
                x[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

void ifft(std::vector<std::complex<float>>& x) {
    size_t n = x.size();

    // conjugate input
    for (auto& c : x) c = std::conj(c);

    // run forward fft
    fft(x);

    // conjugate again and divide by n
    for (auto& c : x) {
        c = std::conj(c) / static_cast<float>(n);
    }
}

std::vector<Tone> generateTones(std::vector<std::complex<float>> dftResult) {
    std::vector<Tone> tones;

    int N = dftResult.size();

    for(int k = 0; k < N / 2; ++k) { // hermitian symmetry
        Tone t;
        t.frequency = k * SAMPLE_RATE / N;
        t.amplitude = std::abs(dftResult[k]) / N;
        t.phase = std::arg(dftResult[k]);
        tones.push_back(t);
    }
    return tones;
}

void sortTones(std::vector<Tone>& tones){
    std::sort(tones.begin(), tones.end(), [](const Tone& a, const Tone& b) {
        return a.amplitude > b.amplitude; 
    });
}

// void synthesizeTones(const std::vector<Tone>& tones, std::vector<std::complex<float>>& buffer) {
//     int N = buffer.size();
//     for (size_t i = 0; i < NUM_TONES; ++i) {
//         Tone t = tones[i];
//         for (int n = 0; n < N; ++n) {
//             float t = 2.0f * M_PI * (float)n * t.frequency / SAMPLE_RATE + t.phase;
//         }
//     }
// }

void generatorThread() {

    std::vector<float> audioSamples;
    loadWav(AUDIO_PATH, audioSamples);

    void* pWriteBuffer;

    size_t index = 0;
    std::vector<float> audioBuffer(WINDOW_SIZE + HOP_SIZE, 0); // overlap add buffer (1024 + 512)

    std::vector<float> hannWindow(WINDOW_SIZE);
    // precalc hann window
    for (int i = 0; i < WINDOW_SIZE; ++i) {
        hannWindow[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / WINDOW_SIZE));
    }

    std::vector<std::complex<float>> buffer(WINDOW_SIZE);
    std::vector<std::complex<float>> outputBuffer(WINDOW_SIZE);
    std::vector<Tone> tones(WINDOW_SIZE);

    g_visualBuffers[0].resize(WINDOW_SIZE / 2);
    g_visualBuffers[1].resize(WINDOW_SIZE / 2);
    
    while(keepRunning){

        ma_uint32 availableSpace = ma_pcm_rb_available_write(&g_rb);
        
        if (availableSpace >= HOP_SIZE) {

            ma_uint32 framesWritten = 0;
            while (framesWritten < HOP_SIZE) {
                ma_uint32 framesToAcquire = HOP_SIZE - framesWritten;
                void* pWriteBuffer;
                
                ma_pcm_rb_acquire_write(&g_rb, &framesToAcquire, &pWriteBuffer);

                float* out = static_cast<float*>(pWriteBuffer);
                for (ma_uint32 i = 0; i < framesToAcquire; ++i) {
                    out[i] = audioBuffer[framesWritten + i]; 
                }

                ma_pcm_rb_commit_write(&g_rb, framesToAcquire);
                framesWritten += framesToAcquire;
            }

            // shift buffer left by HOP_SIZE frames
            std::move(audioBuffer.begin() + HOP_SIZE, audioBuffer.end(), audioBuffer.begin());

            std::fill(audioBuffer.begin() + WINDOW_SIZE, audioBuffer.end(), 0.0f);

            for(int i = 0; i < WINDOW_SIZE; ++i) {
                float v = audioSamples[index + HOP_SIZE + i] * hannWindow[i];
                buffer[i] = std::complex<float>(v, 0.0f);
            }

                auto a = std::chrono::high_resolution_clock::now();

            // fft
            fft(buffer);

                auto b = std::chrono::high_resolution_clock::now();
            
            // tones
            tones = generateTones(buffer);
            sortTones(tones);
            int backBufferIdx = 1 - g_latestBufferIdx.load(std::memory_order_relaxed);
            g_visualBuffers[backBufferIdx] = tones;
            g_latestBufferIdx.store(backBufferIdx, std::memory_order_release);

                auto c = std::chrono::high_resolution_clock::now();
            
            // ifft
            ifft(buffer);

            // instead of running the ifft on the buffer, try synthesizing using the array of tones instead
            // synthesizeTones(tones, outputBuffer);


                auto d = std::chrono::high_resolution_clock::now();

            // update buffer
            for (int i = 0; i < WINDOW_SIZE; ++i) {
                audioBuffer[HOP_SIZE + i] += buffer[i].real();
            }

            // std::cout <<
            //     " | DFT time: " << std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() * 0.001 << " ms" <<
            //     " | Tone time: " << std::chrono::duration_cast<std::chrono::microseconds>(c - b).count() * 0.001 << " ms" <<
            //     " | IDFT time: " << std::chrono::duration_cast<std::chrono::microseconds>(d - c).count() * 0.001 << " ms" <<
            // std::endl;

            index += HOP_SIZE;
            if(index + WINDOW_SIZE + HOP_SIZE > audioSamples.size()){
                index = 0;
            }

        } else {
            ma_pcm_rb_commit_write(&g_rb, 0);
            ma_sleep(2); 
        }
       
    }
}

// jemini <3
void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    ma_uint32 framesTotalRead = 0;
    ma_uint32 bytesPerFrame = ma_get_bytes_per_frame(pDevice->playback.format, pDevice->playback.channels);
    ma_uint8* pOutCursor = static_cast<ma_uint8*>(pOutput);

    // 3. Loop to handle ring buffer wrap-around
    while (framesTotalRead < frameCount) {
        ma_uint32 framesToRead = frameCount - framesTotalRead;
        void* pReadBuffer;

        ma_pcm_rb_acquire_read(&g_rb, &framesToRead, &pReadBuffer);

        if (framesToRead == 0) {
            break; // Buffer underflow (generator thread isn't keeping up)
        }
        memcpy(pOutCursor, pReadBuffer, framesToRead * bytesPerFrame);
        ma_pcm_rb_commit_read(&g_rb, framesToRead);

        pOutCursor += framesToRead * bytesPerFrame;
        framesTotalRead += framesToRead;
    }

    // If we broke out of the loop early (underflow), fill the remaining buffer with silence
    if (framesTotalRead < frameCount) {
        ma_uint32 framesRemaining = frameCount - framesTotalRead;
        memset(pOutCursor, 0, framesRemaining * bytesPerFrame);
    }
}


int main() {
    SetTraceLogLevel(LOG_NONE);
    InitWindow(800, 600, "dsp");
    SetTargetFPS(60);

    if (ma_pcm_rb_init(FORMAT, CHANNELS, RING_BUFFER_SIZE, nullptr, nullptr, &g_rb) != MA_SUCCESS) {
        std::cout << "Failed to initialize ring buffer." << std::endl;
        return -1;
    }

    ma_device_config deviceConfig = ma_device_config_init(ma_device_type_playback);
    deviceConfig.playback.format   = FORMAT;
    deviceConfig.playback.channels = CHANNELS;
    deviceConfig.sampleRate        = SAMPLE_RATE;
    deviceConfig.dataCallback      = data_callback;

    ma_device device;
    if (ma_device_init(NULL, &deviceConfig, &device) != MA_SUCCESS) {
        std::cout << "Failed to initialize playback device." << std::endl;
        ma_pcm_rb_uninit(&g_rb);
        return -1;
    }

    std::thread genThread(generatorThread);

    std::cout << "Pre-rolling buffer..." << std::endl; // remove ts
    while(ma_pcm_rb_available_read(&g_rb) < RING_BUFFER_SIZE) {
        ma_sleep(10); 
    }

    ma_device_start(&device);

    Color bgcol = { 13, 14, 18, 255 };
    while(!WindowShouldClose()){

        int readIdx = g_latestBufferIdx.load(std::memory_order_acquire);
        const auto& tones = g_visualBuffers[readIdx];

        std::vector<float> drawBuffer(800, 0.0f);
        for(int i = 0; i < drawBuffer.size(); i++){
            for(int k = 0; k < NUM_TONES; k++){
                Tone t = tones.at(k);
                float n = i/((float)drawBuffer.size()) * t.frequency * 2.0f * M_PI * (WINDOW_SIZE * 1e-5f * 2.25f);
                float amplitude = std::log10(t.amplitude + 1.0f) * 4.0f;
                float sample = amplitude * sinf((float)t.phase + n);
                drawBuffer[i] += sample;
            }
        }
        BeginDrawing();
        ClearBackground(bgcol);
        
        for(int i = 0; i < drawBuffer.size()-1; i++){
            float a = drawBuffer[i] * 200.0f;
            float b = drawBuffer[i+1] * 200.0f;
            float xScale = ((float)GetScreenWidth()) / drawBuffer.size();
            DrawLine(i * xScale, 300 + a, (i+1) * xScale, 300 + b, GREEN);
        }

        // DrawFPS(10, 770);

        EndDrawing();

    }

    keepRunning = false;
    genThread.join(); 

    ma_device_uninit(&device);
    ma_pcm_rb_uninit(&g_rb);

    return 0;
}