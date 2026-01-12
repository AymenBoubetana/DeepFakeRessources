# InterfaceFlask — DeepFake Detector

Lightweight Flask web UI for a DeepFake detection project combining a video model and an audio model. Upload a video via the web interface to get a combined fake/real prediction (70% video / 30% audio).

## Repo layout
- app.py — Flask app and model loading/inference
- templates/index.html — web UI
- static/uploads/ — uploaded videos
- README.md — this file

## Requirements
- Python 3.8+
- PyTorch (CUDA optional)
- torchvision
- OpenCV (cv2)
- moviepy
- librosa
- whisper
- numpy

Install typical deps:
pip install -r requirements.txt
(Or install packages listed above individually.)

## Running
1. Place the pretrained model files (see Models) into the project root.
2. Start the app:
python app.py
3. Open http://127.0.0.1:5000 and upload a video.

## Models
This project expects two pretrained model files:
- Video model: best_model.pt
- Audio model: audio_model.pth

Both models currently exist in Drive. Update the link below to point to the actual location:
Models download / location: [Drive — modify this link](DRIVE_LINK_TO_MODELS_HERE)

After downloading, place the files in the project root (same folder as app.py) or update app.py paths accordingly.

## Notes
- Device selection: the app uses CUDA if available.
- Audio is extracted via MoviePy and preprocessed with Whisper utilities.
- The video pipeline samples 45 frames resized to 224×224 and runs through an EfficientNet+LSTM+attention architecture.
- The output is a weighted score; final label is "FAKE" if final >= 0.5.

## Troubleshooting
- If model loading fails due to state-dict key mismatches, app.py contains non-strict loading and key remapping logic — ensure model versions match the architectures defined.
- If audio extraction fails, verify ffmpeg is installed and available in PATH.

## License
Describe your license here (e.g., MIT) or remove if not applicable.

<!-- Replace DRIVE_LINK_TO_MODELS_HERE with your actual link -->