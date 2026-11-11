# ExecuTorch Showreel

An Android app that runs six models with [ExecuTorch](https://github.com/pytorch/executorch), on-device, one after another. A vision model gets the screen for 7 seconds and a text model up to 40 seconds, and each prints its measured latency. It is my own project, not an ExecuTorch sample.

| Act | Model | What runs |
|---|---|---|
| Object detection | RT-DETRv2-S (COCO) | boxes over the live frame |
| Semantic segmentation | PIDNet-S (Cityscapes) | 19-class palette over the live frame |
| Monocular depth | Depth Anything V2-S | colour ramp, near red to far blue |
| Background removal | MODNet | alpha matte cut live |
| Text generation | Qwen3.5-0.8B, 8da4w | tokens streamed from a fixed prompt |
| Speech recognition | Whisper-tiny | a WAV transcribed, token by token |

## How it is built

A vision act goes through these steps:

- **Preprocessing:** `org.pytorch.executorch.extension.image.ImageProcessor`, fed from CameraX's YUV planes with `processYuv`, so the model's input doesn't go through a Bitmap.
- **The model:** a `.pte` file per act, which is ExecuTorch's model format.
- **Postprocessing:** [executorch-vision](https://github.com/john-rocky/executorch-vision) for the decoders (DETR, multi-class segmentation, depth, matting) and the drawing.

The two text acts use `LlmModule` and `AsrModule` instead. [`Act.kt`](app/src/main/java/io/github/johnrocky/etshowreel/Act.kt) holds one class per model: its preprocessing config and what it draws. [`MainActivity.kt`](app/src/main/java/io/github/johnrocky/etshowreel/MainActivity.kt) runs the acts in order and switches between the camera and the text screens.

## Running it

Before you build, you need:

- An ExecuTorch Android AAR in `app/libs/`. I built mine from the branch of [#21830](https://github.com/pytorch/executorch/pull/21830), which added `ImageProcessor`. That change has been in ExecuTorch releases since 1.5.0.
- [executorch-vision](https://github.com/john-rocky/executorch-vision), now archived, cloned next to this repository's folder, because `app/build.gradle` compiles its sources.

The app needs a 64-bit phone on Android 9 or later, and it asks for the camera when it opens.

```bash
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

The `.pte` files come from [executorch-models](https://github.com/john-rocky/executorch-samples/tree/main/executorch-models), which links each model's Hugging Face repository. The Whisper files come from [Optimum-ExecuTorch](https://github.com/huggingface/optimum-executorch), which produces the single `.pte` with `encoder` and `text_decoder` that `AsrModule` requires. Run these with `optimum-cli` installed and an ExecuTorch checkout in the current folder:

```bash
optimum-cli export executorch --model openai/whisper-tiny \
    --task automatic-speech-recognition --recipe xnnpack --output_dir whisper
python executorch/extension/audio/mel_spectrogram.py \
    --output_file whisper/preprocess.pte --feature_size 80 --max_audio_len 60
```

The Qwen `.pte` repository on Hugging Face has no `tokenizer.json`, and `optimum-cli` doesn't write one for Whisper, so bring the ones for Qwen3.5 0.8B and Whisper-tiny. `speech.wav` is a 16 kHz mono WAV, 16-bit or 32-bit. The caption always says 8.0 s, the length of my clip. Push them all into the app's external files directory, under these names:

```bash
D=/sdcard/Android/data/io.github.johnrocky.etshowreel/files
adb push rtdetrv2_s_r18vd_xnnpack_fp32.pte        $D/detect.pte
adb push pidnet_s_cityscapes_xnnpack_fp32.pte     $D/segment.pte
adb push depth_anything_v2_small_xnnpack_fp16.pte $D/depth.pte
adb push modnet_portrait_matting_xnnpack_fp16.pte $D/matte.pte
adb push qwen3_5_0_8b_xnnpack_8da4w_e8.pte        $D/model.pte
adb push qwen3.5-0.8b/tokenizer.json              $D/tokenizer.json
adb push whisper/model.pte                        $D/asr.pte
adb push whisper/preprocess.pte                   $D/asr_preprocess.pte
adb push whisper/tokenizer.json                   $D/asr_tokenizer.json
adb push speech.wav                               $D/speech.wav
adb shell "chmod 666 $D/*"
```

An act whose file is missing says so on screen and the app moves on, so you can start with a few of them.

### Recording it

To record the same clip on every run, push a `scene.mp4` to the same directory, before the `chmod` above. The app then plays it instead of the camera:

```bash
adb push scene.mp4 $D/scene.mp4
```

Each act starts at its own offset in the clip (`sceneOffsetSeconds` in `Act.kt`). MODNet needs a person in frame, and on an empty pavement it finds no foreground. My clip, which isn't in this repo, is a 21-second pan across a street followed by 7 seconds of a portrait, so the matting act starts at 21.

## Problems I ran into

**The chroma plane is one byte short.** CameraX's `planes[1]` and `planes[2]` are views one byte apart into the same interleaved allocation, so whichever one you pass ends before its last chroma pair. A bounds check derived from `uvStride * (height / 2 - 1) + width` rejects every real frame. The fix went into ExecuTorch with [#21830](https://github.com/pytorch/executorch/pull/21830).

**Loading the large models is slow.** The LLM is 647 MB and Whisper 230 MB. The app creates both modules on a background thread at startup, while the camera acts run, and each loads its weights the first time it is used.

**Only one vision model stays open at a time.** XNNPACK's Android preset turns on the shared workspace, so every delegate instance across every model uses one arena. When I held four open and ran them in turn, the app crashed. A model that initialized before a larger one grew the arena writes past the arena's end, and the crash comes later, during inference.

**The native loader is initialized once, on the main thread.** ExecuTorch's classes check whether the native library is initialized and then initialize it, with no lock. Opening a vision model on one thread while the LLM opened on another lost that race and threw.

## Licence

BSD-3-Clause, matching ExecuTorch. Each model has its own licence, on its Hugging Face card.
