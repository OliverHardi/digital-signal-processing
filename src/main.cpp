#include <iostream>
#include <vector>
#include <complex>
#include <thread>
#include <atomic>
#include <cmath>
#include <chrono>


#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

constexpr int SAMPLE_RATE = 44100;
constexpr int CHANNELS = 1;
constexpr ma_format FORMAT = ma_format_f32;

constexpr int WINDOW_SIZE = 1 << 10; // 2^10=1024
constexpr int HOP_SIZE = WINDOW_SIZE / 2;

std::atomic<bool> keepRunning{true};

ma_pcm_rb g_rb; // global ring buffer

struct Tone{
    float frequency;
    float amplitude;
    float phase;
};

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

    for(int k = 0; k < N / 2; ++k) {
        Tone t;
        t.frequency = k * SAMPLE_RATE / N;
        t.amplitude = std::abs(dftResult[k]) / N;
        t.phase = std::arg(dftResult[k]);
        tones.push_back(t);
    }
    return tones;
}

void generatorThread() {

    std::vector<float> audioSamples;
    loadWav("zaudio/eyesight.wav", audioSamples);

    void* pWriteBuffer;

    size_t index = 0;
    std::vector<float> audioBuffer(WINDOW_SIZE + HOP_SIZE, 0); // overlap add buffer (1024 + 512)

    std::vector<float> hannWindow(WINDOW_SIZE);
    // precalc hann window
    for (int i = 0; i < WINDOW_SIZE; ++i) {
        hannWindow[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / WINDOW_SIZE));
    }

    // std::vector<std::complex<float>> dftResult(WINDOW_SIZE);
    std::vector<std::complex<float>> buffer(WINDOW_SIZE);
    std::vector<Tone> tones(WINDOW_SIZE);
    
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

            fft(buffer);

                auto b = std::chrono::high_resolution_clock::now();

            tones = generateTones(buffer);

                auto c = std::chrono::high_resolution_clock::now();

            ifft(buffer);

                auto d = std::chrono::high_resolution_clock::now();


            for (int i = 0; i < WINDOW_SIZE; ++i) {
                audioBuffer[HOP_SIZE + i] += buffer[i].real();
            }

            // temp fill in with sin wave
            // for (int i = 0; i < WINDOW_SIZE; ++i) {
            //     audioBuffer[HOP_SIZE + i] += 0.5f * std::sin(2.0f * M_PI * 440.0f * (i) / SAMPLE_RATE) * hannWindow[i];
            // }

            std::cout <<
                " | DFT time: " << std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() * 0.001 << " ms" <<
                " | Tone time: " << std::chrono::duration_cast<std::chrono::microseconds>(c - b).count() * 0.001 << " ms" <<
                " | IDFT time: " << std::chrono::duration_cast<std::chrono::microseconds>(d - c).count() * 0.001 << " ms" <<
            std::endl;
            
            // increase buffer size - done
            // preallocate vectors  - done  
            // n/2 dft size
            // sin/cos twiddle factors


            
            
            
            index += HOP_SIZE;

        } else {
            ma_pcm_rb_commit_write(&g_rb, 0);
            ma_sleep(2); 
        }
       
    }
}

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

    if (ma_pcm_rb_init(FORMAT, CHANNELS, 16384, nullptr, nullptr, &g_rb) != MA_SUCCESS) {
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

    std::cout << "Pre-rolling buffer..." << std::endl;
    while(ma_pcm_rb_available_read(&g_rb) < 8192) {
        ma_sleep(10); 
    }

    ma_device_start(&device);

    std::cout << "enter to exit" << std::endl;
    getchar();

    keepRunning = false;
    genThread.join(); 

    ma_device_uninit(&device);
    ma_pcm_rb_uninit(&g_rb);

    return 0;
}