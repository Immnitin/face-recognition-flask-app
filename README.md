---
title: Face Recognition App
emoji: 📉
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
---
title: Face Recognition App
emoji: 👤
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# Face Recognition App

A Flask-based face recognition application with enrollment and validation features.

## Features
- Real-time face detection and validation
- Multi-pose enrollment (neutral, left, right, blink)
- Face quality checks (blur, exposure)
- User gallery for viewing enrolled images
- Face embedding storage using DeepFace

## Usage
1. Enter a User ID to start enrollment
2. Follow the prompts for different poses
3. View your enrolled images in the gallery

The app uses MediaPipe for face detection and DeepFace for face recognition.
