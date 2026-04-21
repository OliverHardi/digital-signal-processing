#ifndef AUDIO_ENGINE_H
#define AUDIO_ENGINE_H

#include <vector>
#include <mutex>

#include "miniaudio.h"
#include "dsp.h"

class AudioEngine{
public:
    AudioEngine(const char* filename, int sampleRate, int binSize);
    ~AudioEngine();
    
    int sampleRate;
    std::vector<float> fullAudioBuffer;

    std::vector<Tone> tones;
    std::mutex toneMutex;

    void updateTones(const std::vector<Tone>& newTones);

private:
    ma_decoder decoder;
    ma_device device;

    

    static void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount);

};

#endif