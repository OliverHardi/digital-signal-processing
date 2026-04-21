#ifndef DSP_H
#define DSP_H

#include <vector>
#include <complex>

struct Tone {
    float frequency;
    float amplitude;
    float targetAmplitude;
    double phase;
};

std::vector<double> generateHannWindow(size_t N);

std::vector<std::complex<double>> DFT(const std::vector<float>& samples, size_t start, size_t end);

void processOutput(const std::vector<std::complex<double>>& outputs, std::vector<Tone>& tones, int binSize, int sampleRate);

void sortTones(std::vector<Tone>& tones);

float scaleAmplitude(float x);

#endif