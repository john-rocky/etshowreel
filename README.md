# ExecuTorch Showreel

Six models, one Android app, one screen each. Vision models run on the live camera;
the LLM and the speech model take the screen over and stream text. Everything runs
on the device — no network, no server, no account.

| Act | Model | What runs |
|---|---|---|
| Object detection | RT-DETRv2-S (COCO) | boxes over the live frame |
| Semantic segmentation | PIDNet-S (Cityscapes) | 19-class palette over the live frame |
| Monocular depth | Depth Anything V2-S | colour ramp, near red to far blue |
| Background removal | MODNet | alpha matte cut live |
| Text generation | Qwen3.5-0.8B, 8da4w | tokens streamed from a fixed prompt |
| Speech recognition | Whisper-tiny | a WAV transcribed, token by token |

Each act prints its own measured latency, so a recording of the app is its own
evidence.

## What this is for

It is a demo of the three pieces an ExecuTorch app actually needs, and of where
each of them comes from:

- **Preprocessing** — `org.pytorch.executorch.extension.image.ImageProcessor`,
  fed straight from CameraX's YUV planes with `processYuv`. No Bitmap round-trip
  per frame.
- **The model** — a `.pte` per act, plus `LlmModule` and `AsrModule` for the two
  runners.
- **Postprocessing** — [executorch-vision](https://github.com/john-rocky/executorch-vision)
  for the decoders (DETR, multi-class segmentation, depth, matting) and the
  drawing.

`Act.kt` holds one class per model: its preprocessing config and what it draws.
`MainActivity.kt` is the reel — a clock, a camera path, and a text path.

## Running it

```bash
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Then push the models into the app's external files directory. They come from the
[executorch-models](https://github.com/john-rocky/executorch-models) shelf:

```bash
D=/sdcard/Android/data/io.github.johnrocky.etshowreel/files
adb push rtdetrv2_s_r18vd_xnnpack_fp32.pte        $D/detect.pte
adb push pidnet_s_cityscapes_xnnpack_fp32.pte     $D/segment.pte
adb push depth_anything_v2_small_xnnpack_fp16.pte $D/depth.pte
adb push modnet_portrait_matting_xnnpack_fp16.pte $D/matte.pte
adb push qwen3_5_0_8b_8da4w/model.pte             $D/model.pte
adb push qwen3_5_0_8b_8da4w/tokenizer.json        $D/tokenizer.json
adb push whisper/model.pte                        $D/asr.pte
adb push whisper/preprocess.pte                   $D/asr_preprocess.pte
adb push whisper/tokenizer.json                   $D/asr_tokenizer.json
adb push speech.wav                               $D/speech.wav
adb shell "chmod 666 $D/*"
```

An act whose file is missing says so on screen and the reel moves on, so a subset
works fine.

### Recording it

A recording made on live frames shows whatever the lens happened to be pointed
at. Drop a `scene.mp4` in the same directory and the reel plays that instead of
the camera, so the same run produces the same frames every time:

```bash
adb push scene.mp4 $D/scene.mp4
```

Each act names its own start offset into the clip (`sceneOffsetSeconds` in
`Act.kt`), because which model wants which footage is not a matter of taste —
MODNet pointed at an empty pavement correctly reports no foreground at all. The
bundled clip used for the demo is a 21-second pan across a street followed by
7 seconds of a portrait; the matting act starts at 21.

The Whisper pair comes from
[Optimum-ExecuTorch](https://github.com/huggingface/optimum-executorch), which
produces the single `.pte` exposing `encoder` and `text_decoder` that `AsrModule`
requires:

```bash
optimum-cli export executorch --model openai/whisper-tiny \
    --task automatic-speech-recognition --recipe xnnpack --output_dir whisper
python executorch/extension/audio/mel_spectrogram.py \
    --output_file whisper/preprocess.pte --feature_size 80 --max_audio_len 60
```

`speech.wav` is any 16 kHz mono WAV.

## Two things worth knowing

**The chroma plane is one byte short.** CameraX's `planes[1]` and `planes[2]` are
views one byte apart into the same interleaved allocation, so whichever one you
pass ends before its last chroma pair. A bounds check derived from
`uvStride * (height / 2 - 1) + width` rejects every real frame; the fix is
upstream in
[pytorch/executorch#21830](https://github.com/pytorch/executorch/pull/21830).

**Loading is the long pole, not inference.** 620 MB of LLM weights and 230 MB of
Whisper take long enough to open that the reel maps both on a background thread at
startup, behind the camera acts.

**Only one vision model stays open at a time.** XNNPACK's Android preset turns on
the shared workspace, so every delegate instance across every model uses one
arena. Holding four open and running them in turn faults: whichever model
initialized before a larger one grew the arena writes past its end, and the crash
lands in the inference rather than at the resize that caused it.

**The native loader is initialized once, on the main thread.** `Module`,
`LlmModule` and `AsrModule` each start with "if not initialized, then
initialize", with nothing between the check and the act. Opening a vision model
on one thread while the LLM opens on another loses that race and throws.

## Licence

BSD-3-Clause, matching ExecuTorch. The models carry their own licences — see the
model cards on the shelf.
