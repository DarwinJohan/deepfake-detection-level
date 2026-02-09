## Deepfake Detection Framework (6-Level)

This system is designed to detect deepfake videos in a tiered approach, from basic biological signals to advanced texture and audio analysis. Each level adds an extra verification layer, making detection more robust.

## Level 1: Emotion Detection 😃😢😡

Goal: Detect natural facial expressions to find emotional inconsistencies.

Method: DeepFace or any facial expression recognition model.

Limitation: Some deepfakes mimic expressions → further levels required.

Output: Emotion histogram per frame or video.

## Level 2: Blink Detection 👁️

Goal: Detect natural blinking patterns since early deepfakes often forget to blink.

Method: OpenCV Haar Cascade, dynamic EAR approximation, smoothing.

Output: Blink count, blink rate, suspicious flag.

## Level 3: Head Pose Consistency

Goal: Check if head rotations (yaw/pitch/roll) match natural facial movements.

Method: Landmark tracking via Mediapipe/dlib or Haar Cascade approximation.

Output: Head rotation pattern vs eye/mouth movement.

Suspicious indicator: Head moves but eyes/mouth remain static.

## Level 4: Texture & Noise Analysis 🎨

Goal: Detect texture artifacts, blur, or unnatural noise patterns.

Method:

Local Binary Patterns (LBP)

FFT / DCT analysis

Output: Heatmap per frame and anomaly score.

## Level 5: Color & Lighting Consistency 🌈💡

Goal: Check for consistency in skin tone and lighting across frames.

Method: HSV / LAB statistics, frame-to-frame variation.

Output: Warnings if tone or lighting suddenly changes.

## Level 6: Lip Sync / Audio-Visual Consistency 🎤👄

Goal: Detect mismatch between mouth movements and audio.

Method:

Extract phonemes from audio

Track MAR (Mouth Aspect Ratio) from video

Compute correlation → mismatch indicates suspicion

Output: Lip-sync score, deepfake probability.


## Framework Overview

Here’s the overview of our 6-level deepfake detection system:

![Framework Overview](assets/framework.png)
