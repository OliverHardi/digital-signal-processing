#include "audio_engine.h"
#include <iostream>

AudioEngine::AudioEngine(const char* filename, int sampleRate, int binSize) : sampleRate(sampleRate){
    ma_decoder_config decoderConfig = ma_decoder_config_init(ma_format_f32, 1, sampleRate);
    if (ma_decoder_init_file(filename, &decoderConfig, &decoder) != MA_SUCCESS) {
        std::cerr << "Failed to load: " << filename << std::endl;
        return;
    }

    ma_uint64 totalFrames;
    ma_decoder_get_length_in_pcm_frames(&decoder, &totalFrames);
    fullAudioBuffer.resize(totalFrames);
    ma_decoder_read_pcm_frames(&decoder, fullAudioBuffer.data(), totalFrames, NULL);

    ma_device_config config = ma_device_config_init(ma_device_type_playback);
    config.playback.format   = ma_format_f32;
    config.playback.channels = 1;
    config.sampleRate        = sampleRate;
    config.dataCallback      = data_callback;
    config.pUserData         = this;

    if (ma_device_init(NULL, &config, &device) != MA_SUCCESS) {
        std::cerr << "Failed to open playback device." << std::endl;
        ma_decoder_uninit(&decoder);
        return;
    }

    ma_device_start(&device);
}

AudioEngine::~AudioEngine() {
    ma_device_uninit(&device);
    ma_decoder_uninit(&decoder);
}

void AudioEngine::data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    AudioEngine* pEngine = (AudioEngine*)pDevice->pUserData;
    if (pEngine == nullptr || pOutput == nullptr) return;

    float* pOut = (float*)pOutput;

    for(ma_uint32 i = 0; i < frameCount; ++i){ pOut[i] = 0.0f; }

    // lock tones while reading
    std::lock_guard<std::mutex> lock(pEngine->toneMutex);

    for(size_t k = 0; k < pEngine->tones.size(); k++){ // every tone in the array
        Tone& t = pEngine->tones[k];
        // if (t.amplitude < 0.001f) continue;

        for (ma_uint32 i = 0; i < frameCount; ++i) {
            float sample = t.amplitude * sinf((float)t.phase);
            pOut[i] += sample;

            t.phase += 2.0f * M_PI * t.frequency / pEngine->sampleRate;
            if (t.phase > 2.0f * M_PI) t.phase -= 2.0f * M_PI;
        }
    }

    // prevent clipping maybe

}

void AudioEngine::updateTones(const std::vector<Tone>& newTones){
    std::lock_guard<std::mutex> lock(toneMutex);
    // tones = newTones;
    if (tones.size() != newTones.size()) {
        tones = newTones;
        return;
    }

    for (size_t i = 0; i < newTones.size(); i++) {
        tones[i].frequency = newTones[i].frequency;
        tones[i].amplitude = newTones[i].amplitude;
    }
    
    // for (const Tone& incoming : newTones) {
    //     bool matched = false;
    //     for (Tone& existing : tones) {
    //         if (fabs(existing.frequency - incoming.frequency) < 1.0f) {
    //             existing.amplitude = incoming.amplitude; // smooth this too ideally
    //             matched = true;
    //             break;
    //         }
    //     }
    //     if (!matched) {
    //         tones.push_back(incoming); // new tone, phase starts at whatever DFT gave
    //     }
    // }
}