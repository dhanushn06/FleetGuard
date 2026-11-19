# Signal Quality Intelligence & Reliability Gate Documentation

## Overview
PulseCam FleetGuard employs a multi-stage, rule-based Signal Quality Intelligence Gate designed by **Member 3**. The system enforces deterministic checks on raw camera signals, temporal tracking, multi-region illumination, and cross-algorithm agreement before allowing an estimated heart rate to be accepted.

---

## Documented Criteria & Thresholds

| Criterion | Threshold | Measurement Domain | Physical / Optical Justification |
| :--- | :--- | :--- | :--- |
| **Minimum Duration** | $\ge 10.0\text{ s}$ | Time base ($t$) | Required for Welch PSD frequency resolution ($\approx 0.125\text{ Hz}$ bin width with $8\text{ s}$ segment) to separate physiological cardiac harmonics from low-frequency baseline drift. |
| **Tracking Fraction** | $\ge 80\%$ ($0.80$) | MediaPipe track flag | Prevents excessive spline interpolation across dropped frames, which could synthesize artificial periodic oscillations. |
| **Mean Motion** | $\le 5.0\text{ px}$ | Landmark centroid displacement | High head movement shifts facial pixel boundaries across shadows and background, introducing Doppler and illumination artifacts. |
| **Peak Motion** | $\le 15.0\text{ px}$ | Landmark centroid displacement | Rejects sudden head turns or gestures that disrupt blood volume pulse (BVP) extraction. |
| **Minimum Brightness** | $\ge 30.0$ (8-bit) | Regional pixel mean ($Y$) | In low-light environments, CMOS camera sensor shot noise and quantization error obscure small optical rPPG modulation ($\approx 0.5 - 2\%$). |
| **Maximum Brightness** | $\le 230.0$ (8-bit) | Regional pixel mean ($Y$) | Severe overexposure and specular reflections saturate camera color channels, clipping the pulsatile waveform. |
| **Regional Spectral SNR** | $\ge 0.0\text{ dB}$ | Welch PSD ($0.75 - 3.0\text{ Hz}$) | The cardiac peak power within a $\pm 0.1\text{ Hz}$ window must exceed the in-band noise floor power. Negative SNR indicates noise dominance. |
| **Spectral Concentration** | $\ge 0.25$ | Welch PSD ($0.75 - 3.0\text{ Hz}$) | At least 25% of total in-band energy must be focused within $\pm 0.2\text{ Hz}$ of the cardiac peak, filtering out broadband noise. |
| **Minimum Valid Regions** | $\ge 2\text{ of } 8$ | Facial ROI mesh | Cross-region spatial verification requires at least two distinct anatomical sites to confirm physiological synchrony. |
| **Algorithm Agreement** | $\le 8.0\text{ BPM}$ | Green vs CHROM vs POS | If classical estimators diverge by $>8\text{ BPM}$, the signal is contaminated by non-pulsatile color distortion. |
| **Regional Agreement** | $\le 10.0\text{ BPM}$ | Across accepted ROIs | Human facial perfusion is temporally synchronized; divergence across regions indicates localized motion or shadows. |

---

## Explainable Composite Quality Score Formulation

When a measurement is accepted, an explainable composite quality score $Q \in [0.0, 1.0]$ is computed:

$$Q = 0.25 \cdot S_{\text{track}} + 0.25 \cdot S_{\text{motion}} + 0.25 \cdot S_{\text{bright}} + 0.25 \cdot S_{\text{SNR}}$$

Where:
- $S_{\text{track}} = \text{tracking\_fraction} \in [0, 1]$
- $S_{\text{motion}} = \max\left(0, 1 - \frac{\text{mean\_motion}}{5.0}\right)$
- $S_{\text{bright}} = \max\left(0, 1 - \frac{|\text{mean\_brightness} - 130|}{100}\right)$
- $S_{\text{SNR}} = \text{clip}\left(\frac{\overline{\text{SNR}_{\text{selected}}}}{10.0}, 0, 1\right)$

If any mandatory gating criterion fails, the measurement is marked `accepted = False` and `score = None`.

---

## Ethical & Scientific Safeguards
1. **No Fitness-to-Drive Diagnosis**:
   This prototype is an experimental research system. Signal quality and rPPG outputs must never be used for automated employment or fitness-for-duty decisions.
2. **Algorithm Agreement is Not Accuracy Proof**:
   Agreement between Green, CHROM, and POS proves algorithmic self-consistency under current optical conditions, not ground-truth clinical ECG accuracy.
3. **Exploratory Alertness Separation**:
   Visual alertness indicators (Eye Aspect Ratio, Mouth Aspect Ratio, head pose) are processed independently and are labeled strictly exploratory.
