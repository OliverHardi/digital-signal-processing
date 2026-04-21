#include <iostream>
#include <vector>
#include <complex>

#include "raylib.h"

#define SAMPLE_RATE 44100
#define WINDOW_SIZE 1024
#define HOP_SIZE 512
#define N_TONES 1024

struct Tone {
    double frequency;
    double amplitude;
    double phase;
};

void dft(std::vector<Tone>& window, std::vector<float>& samples, size_t start, size_t end){
    std::vector<std::complex<double>> dft(window.size());

    std::cout << "computing: " << start << ", " << end << std::endl;
    // compute DFT using complex

    // convert to vector of tones

    int N = end - start;

    for(int k = 0; k < N; k++){
        std::complex<double> sum(0, 0);
        for(int n = 0; n < N; n++){
            double angle = (2.0 * M_PI * k * n) / N;
            std::complex<double> rotation(cos(angle), -sin(angle));

            double sample = (double)samples[start + n];

            sum += sample * rotation;
        }
        Tone t;
        t.amplitude = (double)k * SAMPLE_RATE / N;
        t.frequency = std::abs(sum) * (2.0 / N);
        t.phase = std::arg(sum);

        window[k] = t;
    }
    
}


int main() {

    InitWindow(800, 600, "Tone Visualizer");
    SetTargetFPS(60);

    std::vector<float> samples(SAMPLE_RATE * 5);

    for(int i = 0; i < samples.size(); i++){
        // 300hz sine wave
        samples[i] = sin(2 * M_PI * 300 * i / SAMPLE_RATE);
    }

    std::vector<Tone> tones(N_TONES);
    dft(tones, samples, 0, WINDOW_SIZE);

    for(int i = 0; i < tones.size(); i++){
        std::cout << "Tone " << i << ": freq=" << tones[i].frequency
                  << ", amp=" << tones[i].amplitude << std::endl;
    }

    // plot tones
    while(!WindowShouldClose()){
        BeginDrawing();
        ClearBackground(RAYWHITE);

        for(int i = 0; i < tones.size(); i++){
            float x = tones[i].frequency / (SAMPLE_RATE / 2) * GetScreenWidth();
            float y = tones[i].amplitude * GetScreenHeight() * 1.0f;
            std::cout << "plotting: " << x << ", " << y << std::endl;
            DrawCircle(x, y, 5, BLACK);
        }
        EndDrawing();
    }
    // std::cout << "done\n";

    return 0;
}