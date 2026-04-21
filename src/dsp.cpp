#include "dsp.h"

std::vector<double> generateHannWindow(size_t N){
    std::vector<double> window(N);
    for(size_t n = 0; n < N; n++){
        window[n] = 0.5f * (1.0f - cosf((2.0f * M_PI * n) / (N - 1)));
    }
    return window;
}

std::vector<std::complex<double>> DFT(const std::vector<float>& samples, size_t start, size_t end){
    int N = end - start;

    std::vector<double> window = generateHannWindow(N);

    std::vector<std::complex<double>> output(N/2 + 1);

    for(int k = 0; k < N / 2; k++){ // hermitian symmetry, only need to compute half of the bin
        std::complex<double> sum(0, 0);
        for(int n = 0; n < N; n++){
            double angle = (2.0 * M_PI * k * n) / N;
            
            std::complex<double> rotation(cos(angle), -sin(angle));

            // double windowedSample = (double)samples[start + n] * window[n];
            double sample = (double)samples[start + n];

            sum += sample * rotation;
        }

        output[k] = sum;
    }

    return output;
}

void processOutput(const std::vector<std::complex<double>>& outputs, std::vector<Tone>& tones, int binSize, int sampleRate){
    tones.clear();

    for(int k = 0; k<binSize/2; k++){
        std::complex<double> output = outputs[k];
        Tone t;

        t.amplitude = std::abs(output) / (binSize);
        t.frequency = k * ((double)sampleRate / binSize);
        t.phase = std::arg(output);
        
        tones.emplace_back(t);
    }
}

void sortTones(std::vector<Tone>& tones){
    std::sort(tones.begin(), tones.end(), [](const Tone& a, const Tone& b) {
        return a.amplitude > b.amplitude;
    });
}

float scaleAmplitude(float x){
    return log10(1.0f + x * 24.0f) * 0.16;
}