#include <Arduino.h>

// ---------------------------------------------------------------------------
// Pin Definitions
// ---------------------------------------------------------------------------
static constexpr uint8_t ECG_PIN  = 34;

// ---------------------------------------------------------------------------
// Sampling Configuration
// ---------------------------------------------------------------------------
static constexpr uint32_t SAMPLE_RATE_HZ     = 250;
static constexpr uint32_t SAMPLE_INTERVAL_US = 1000000UL / SAMPLE_RATE_HZ;

// ---------------------------------------------------------------------------
// Pan-Tompkins Pipeline Configuration
// ---------------------------------------------------------------------------
static constexpr uint8_t  MWI_WINDOW     = 45;      // widened: 180 ms covers full QRS
static constexpr uint8_t  DERIV_LEN      = 5;
static const    int8_t    DERIV_COEFF[5] = {-1, -2, 0, 2, 1};
static constexpr float    THRESHOLD_FRAC = 0.30f;   // lowered: 0.45 was too aggressive
static constexpr float    PEAK_DECAY     = 0.990f;  // faster decay: tracks signal changes quicker
static constexpr uint32_t REFRACTORY_MS  = 300;
static constexpr uint8_t  RR_BUFFER_SIZE = 8;
static constexpr float    BPM_MIN        = 30.0f;
static constexpr float    BPM_MAX        = 220.0f;
static constexpr float    RR_OUTLIER_TOL = 0.25f;   // reject RR intervals >25% from median

// ---------------------------------------------------------------------------
// IIR Bandpass Filter — 2 cascaded biquad sections
// ---------------------------------------------------------------------------
struct Biquad {
    float b0, b1, b2;
    float a1, a2;
    float w1, w2;

    Biquad(float b0, float b1, float b2, float a1, float a2)
        : b0(b0), b1(b1), b2(b2), a1(a1), a2(a2), w1(0.0f), w2(0.0f) {}

    float process(float x) {
        float y = b0 * x + w1;
        w1 = b1 * x - a1 * y + w2;
        w2 = b2 * x - a2 * y;
        return y;
    }
};

// High-pass biquad at ~0.5 Hz to remove baseline wander
static Biquad hpf(0.99556f, -1.99112f, 0.99556f, -1.99112f, 0.99112f);

// Bandpass: 2 cascaded biquads (5–15 Hz passband)
static Biquad bq1(0.13072f,  0.0f, -0.13072f, -1.6278f, 0.7386f);
static Biquad bq2(0.13072f,  0.0f, -0.13072f, -1.8424f, 0.9025f);

// ---------------------------------------------------------------------------
// Filter States
// ---------------------------------------------------------------------------
static float   derivBuf[DERIV_LEN] = {};
static uint8_t derivIdx            = 0;

static float   mwiBuf[MWI_WINDOW] = {};
static uint8_t mwiIdx              = 0;
static float   mwiSum              = 0.0f;

// ---------------------------------------------------------------------------
// R-R Interval Buffer
// ---------------------------------------------------------------------------
static uint32_t rrBuf[RR_BUFFER_SIZE] = {};
static uint8_t  rrIdx                  = 0;
static uint8_t  rrCount                = 0;

// ---------------------------------------------------------------------------
// Algorithm State
// ---------------------------------------------------------------------------
static uint32_t lastSampleUs = 0;
static uint32_t lastBeatMs   = 0;
static float    adaptivePeak = 50.0f;   
static float    threshold    = 15.0f;
static float    mwiPeak      = 0.0f;
static bool     inCandidate  = false;

// ---------------------------------------------------------------------------
// Public Output State (Globals for plotting)
// ---------------------------------------------------------------------------
volatile float   g_instantBPM = 0.0f;
volatile float   g_averageBPM = 0.0f;
volatile float   g_mwiSignal  = 0.0f; // Added for Plotter
volatile float   g_threshold  = 0.0f; // Added for Plotter

// ---------------------------------------------------------------------------
// Math Functions
// ---------------------------------------------------------------------------
static float computeDerivative(float x) {
    derivBuf[derivIdx] = x;
    derivIdx = (derivIdx + 1) % DERIV_LEN;
    float out = 0.0f;
    for (uint8_t i = 0; i < DERIV_LEN; i++) {
        int8_t idx = (int8_t)derivIdx - (int8_t)DERIV_LEN + (int8_t)i;
        if (idx < 0) idx += DERIV_LEN;
        out += (float)DERIV_COEFF[i] * derivBuf[idx];
    }
    return out * 0.125f;
}

static float updateMWI(float x) {
    mwiSum -= mwiBuf[mwiIdx];
    mwiBuf[mwiIdx] = x;
    mwiSum += x;
    mwiIdx = (mwiIdx + 1) % MWI_WINDOW;
    return mwiSum / (float)MWI_WINDOW;
}

static float computeMedianBPM() {
    if (rrCount == 0) return 0.0f;
    uint32_t tmp[RR_BUFFER_SIZE];
    uint8_t  n = min(rrCount, RR_BUFFER_SIZE);
    memcpy(tmp, rrBuf, n * sizeof(uint32_t));

    for (uint8_t i = 1; i < n; i++) {
        uint32_t key = tmp[i];
        int8_t   j   = i - 1;
        while (j >= 0 && tmp[j] > key) {
            tmp[j + 1] = tmp[j];
            j--;
        }
        tmp[j + 1] = key;
    }
    uint32_t medianRR = (n % 2 == 0) ? (tmp[n/2 - 1] + tmp[n/2]) / 2 : tmp[n/2];
    return (medianRR > 0) ? (60000.0f / (float)medianRR) : 0.0f;
}

// ---------------------------------------------------------------------------
// Register Beat
// ---------------------------------------------------------------------------
static void registerBeat(uint32_t nowMs) {
    if (lastBeatMs > 0) {
        uint32_t rr   = nowMs - lastBeatMs;
        float instBPM = 60000.0f / (float)rr;

        if (instBPM >= BPM_MIN && instBPM <= BPM_MAX) {
            if (rrCount >= 4 && g_averageBPM > 0.0f) {
                float medRR = 60000.0f / g_averageBPM;
                float diff  = (float)rr - medRR;
                if (diff < 0) diff = -diff;
                if (diff > medRR * RR_OUTLIER_TOL) {
                    lastBeatMs = nowMs;
                    return;
                }
            }
            g_instantBPM = instBPM;
            rrBuf[rrIdx] = rr;
            rrIdx = (rrIdx + 1) % RR_BUFFER_SIZE;
            if (rrCount < RR_BUFFER_SIZE) rrCount++;
            g_averageBPM = computeMedianBPM();
        }
    }
    lastBeatMs = nowMs;
}

// ---------------------------------------------------------------------------
// Process Sample
// ---------------------------------------------------------------------------
static void processSample(uint16_t rawADC) {
    float x = (float)rawADC - 2048.0f;
    if (x >  2000.0f) x =  2000.0f;
    if (x < -2000.0f) x = -2000.0f;

    float hp         = hpf.process(x);
    float filtered   = bq2.process(bq1.process(hp));
    float deriv      = computeDerivative(filtered);
    float squared    = deriv * deriv;
    float integrated = updateMWI(squared);

    adaptivePeak *= PEAK_DECAY;
    if (integrated > adaptivePeak) adaptivePeak = integrated;
    threshold = THRESHOLD_FRAC * adaptivePeak;

    // Export to globals for plotting
    g_mwiSignal = integrated;
    g_threshold = threshold;

    uint32_t nowMs        = millis();
    bool     inRefractory = (nowMs - lastBeatMs) < REFRACTORY_MS;

    if (!inCandidate) {
        if (!inRefractory && integrated > threshold) {
            inCandidate = true;
            mwiPeak     = integrated;
        }
    } else {
        if (integrated > mwiPeak) {
            mwiPeak = integrated;
        }
        if (integrated < threshold) {
            inCandidate = false;
            if (!inRefractory) {
                registerBeat(nowMs);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Setup & Loop
// ---------------------------------------------------------------------------
void setup() {
    Serial.begin(115200);
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db);
    Serial.println("Raw,MWI,Threshold,AvgBPM"); // Plotter Header
    lastSampleUs = micros();
}

void loop() {
    uint32_t nowUs = micros();
    if ((nowUs - lastSampleUs) < SAMPLE_INTERVAL_US) return;
    lastSampleUs += SAMPLE_INTERVAL_US;

    uint16_t raw = analogRead(ECG_PIN);
    processSample(raw);

    // Formatted for the Arduino Serial Plotter
    Serial.print(raw);
    Serial.print(",");
    Serial.print(g_mwiSignal);
    Serial.print(",");
    Serial.print(g_threshold);
    Serial.print(",");
    Serial.println(g_averageBPM);
}