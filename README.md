Computer Vision Agent
Real-time Image Classification & Object Detection


📌 Overview

This project implements a Computer Vision Agent in Python using:

TensorFlow/Keras (for image classification)
YOLOv8 & OpenCV (for real-time object detection)
The agent can:
✅ Classify images (e.g., CIFAR-10, Cats vs. Dogs)
✅ Detect objects in real-time using a webcam (YOLOv8)
✅ Be extended for custom datasets (e.g., face masks, potholes)

🚀 Features

1. Image Classification (CNN)

Built a Convolutional Neural Network (CNN) from scratch using TensorFlow.
Trained on CIFAR-10 (supports 10 classes: cats, dogs, cars, etc.).
Easily extendable to custom datasets.
2. Object Detection (YOLOv8)

Real-time detection using YOLOv8 (Ultralytics).
Works with images, videos, and webcam feed.
Lightweight (YOLOv8n) or high-accuracy (YOLOv8x) options.
3. Webcam Integration

Smooth OpenCV pipeline for live detection.
Press Q to exit gracefully.
