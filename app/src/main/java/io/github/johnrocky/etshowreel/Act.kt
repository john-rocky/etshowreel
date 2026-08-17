package io.github.johnrocky.etshowreel

import android.graphics.Bitmap
import io.github.johnrocky.etvision.CoordinateMapper
import io.github.johnrocky.etvision.DepthDecoder
import io.github.johnrocky.etvision.Detection
import io.github.johnrocky.etvision.DetrDecoder
import io.github.johnrocky.etvision.MultiClassSegmentationDecoder
import io.github.johnrocky.etvision.ResizeMode
import io.github.johnrocky.etvision.SegmentationDecoder
import io.github.johnrocky.etvision.android.applyAlpha
import io.github.johnrocky.etvision.android.drawDetections
import io.github.johnrocky.etvision.android.overlayLabels
import io.github.johnrocky.etvision.android.toColorMappedBitmap
import java.io.File
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.extension.asr.AsrCallback
import org.pytorch.executorch.extension.asr.AsrModule
import org.pytorch.executorch.extension.asr.AsrTranscribeConfig
import org.pytorch.executorch.extension.image.ImageProcessorConfig
import org.pytorch.executorch.extension.image.Normalization
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmGenerationConfig
import org.pytorch.executorch.extension.llm.LlmModule
import org.pytorch.executorch.extension.llm.LlmModuleConfig

/**
 * One segment of the reel.
 *
 * Two kinds exist because two kinds of model are being shown. A [VisionAct] is pulled along by the
 * camera — it has nothing to do until a frame arrives. A [TextAct] drives itself: it is handed the
 * files directory and streams text out at whatever rate the model produces it.
 */
sealed interface Act {
  /** Shown large on screen; the reel is its own caption. */
  val title: String

  val subtitle: String
}

// ─── vision ───────────────────────────────────────────────────────────────────

/** What a vision act produced: the frame to show and one line naming what happened. */
class Frame(val bitmap: Bitmap, val detail: String)

/**
 * A model fed by the live camera.
 *
 * Each act owns its own [ImageProcessorConfig] because the normalizations genuinely differ —
 * ImageNet, plain 0-1, and [-1, 1] all appear among four models, and using the wrong one produces a
 * plausible-looking wrong answer rather than an error.
 *
 * @property file Name of the `.pte` in the app's external files directory.
 * @property config Preprocessing for this model.
 */
abstract class VisionAct(
    override val title: String,
    override val subtitle: String,
    val file: String,
    val config: ImageProcessorConfig,
) : Act {
  abstract fun render(outputs: Array<EValue>, source: Bitmap, mapper: CoordinateMapper): Frame

  /** Stretch, because every model here was exported without letterboxing. */
  fun mapperFor(source: Bitmap): CoordinateMapper =
      CoordinateMapper.of(
          source.width,
          source.height,
          config.targetWidth,
          config.targetHeight,
          ResizeMode.STRETCH,
      )

  /** Loads this act's module, or null when its file has not been pushed. */
  fun load(dir: File): Module? {
    val f = File(dir, file)
    return if (f.exists()) Module.load(f.absolutePath, Module.LOAD_MODE_MMAP, 0) else null
  }
}

/** RT-DETRv2: boxes drawn over the live frame. */
class Detect :
    VisionAct(
        "Object detection",
        "RT-DETRv2-S · COCO",
        "detect.pte",
        ImageProcessorConfig(
            targetWidth = 640,
            targetHeight = 640,
            normalization = Normalization.zeroToOne(),
        ),
    ) {
  private val decoder = DetrDecoder(scoreThreshold = 0.5f)

  override fun render(outputs: Array<EValue>, source: Bitmap, mapper: CoordinateMapper): Frame {
    val detections: List<Detection> =
        decoder.decode(
            outputs[0].toTensor().dataAsFloatArray,
            outputs[1].toTensor().dataAsFloatArray,
            mapper,
        )
    val names =
        detections.groupingBy { COCO_LABELS[it.classIndex] }.eachCount().entries.joinToString(
            ", ") { "${it.value} ${it.key}" }
    return Frame(
        drawDetections(source, detections, COCO_LABELS),
        if (detections.isEmpty()) "nothing over 50%" else names,
    )
  }
}

/** Depth Anything V2: the colour ramp fills the frame. */
class Depth :
    VisionAct(
        "Monocular depth",
        "Depth Anything V2-S",
        "depth.pte",
        ImageProcessorConfig(
            targetWidth = 518,
            targetHeight = 518,
            normalization = Normalization.imagenet(),
        ),
    ) {
  private val decoder = DepthDecoder()

  override fun render(outputs: Array<EValue>, source: Bitmap, mapper: CoordinateMapper): Frame {
    val map = decoder.decode(outputs[0].toTensor().dataAsFloatArray, 518, 518, mapper)
    return Frame(map.toColorMappedBitmap(), "near is red, far is blue")
  }
}

/** MODNet: the background is cut away live. */
class Matte :
    VisionAct(
        "Background removal",
        "MODNet",
        "matte.pte",
        ImageProcessorConfig(
            targetWidth = 512,
            targetHeight = 512,
            // (pixel / 255 - 0.5) / 0.5 maps 0-255 onto [-1, 1].
            normalization =
                Normalization(
                    1f / 255f,
                    floatArrayOf(0.5f, 0.5f, 0.5f),
                    floatArrayOf(0.5f, 0.5f, 0.5f),
                ),
        ),
    ) {
  // The head already emits probabilities, so no second sigmoid.
  private val decoder = SegmentationDecoder(applySigmoid = false)

  override fun render(outputs: Array<EValue>, source: Bitmap, mapper: CoordinateMapper): Frame {
    val matte = decoder.decode(outputs[0].toTensor().dataAsFloatArray, 512, 512, mapper)
    val covered = matte.values.count { it > 0.5f } * 100f / matte.values.size
    return Frame(applyAlpha(source, matte), "foreground %.0f%%".format(covered))
  }
}

/** PIDNet: the Cityscapes palette over the live frame. */
class Segment :
    VisionAct(
        "Semantic segmentation",
        "PIDNet-S · Cityscapes",
        "segment.pte",
        ImageProcessorConfig(
            targetWidth = 1024,
            targetHeight = 1024,
            normalization = Normalization.imagenet(),
        ),
    ) {
  private val decoder = MultiClassSegmentationDecoder(numClasses = CITYSCAPES_LABELS.size)

  override fun render(outputs: Array<EValue>, source: Bitmap, mapper: CoordinateMapper): Frame {
    val labels = decoder.argmax(outputs[0].toTensor().dataAsFloatArray, 128, 128)
    val scaled = decoder.resizeLabels(labels, 128, 128, source.width, source.height)
    return Frame(
        overlayLabels(source, decoder.colorize(scaled, CITYSCAPES_PALETTE), alpha = 0.55f),
        "19 classes",
    )
  }
}

// ─── text ─────────────────────────────────────────────────────────────────────

/** Where a [TextAct] writes. Every call lands on the UI thread. */
interface TextSink {
  /** Replace the body text. Called on every token, so it has to be cheap. */
  fun body(text: String)

  /** Replace the stats line under the title. */
  fun stats(line: String)

  /** The act has nothing more to say; the reel may advance. */
  fun done()

  /** Report a missing file or a failed load instead of showing an empty screen. */
  fun failed(reason: String)
}

/**
 * A model that produces text on its own clock rather than per camera frame.
 *
 * These two are the heavy loads in the reel — 620 MB of weights for the LLM, 230 MB for Whisper —
 * and mapping them takes long enough to be dead air on screen. [prepare] is called once at startup
 * on a background thread so the load overlaps with the camera acts, and [run] finds the model
 * already open.
 */
abstract class TextAct(override val title: String, override val subtitle: String) : Act {
  /** Open the model. Called off the main thread, before the reel reaches this act. */
  abstract fun prepare(dir: File)

  abstract fun run(dir: File, sink: TextSink)

  /**
   * Cut the act short because the reel's clock ran out.
   *
   * Without this a generation the reel has already left keeps running: it competes with the next
   * act for the same cores, and the cycle after finds the model still busy. The visible symptom is
   * a token rate that gets worse every lap.
   */
  open fun cancel() {}
}

/**
 * Qwen3.5 0.8B at 8da4w: a prompt goes in, tokens stream out.
 *
 * The prompt is fixed so the reel is reproducible — the point being demonstrated is the decode rate
 * on the device, not the model's taste.
 */
class Chat :
    TextAct("Text generation", "Qwen3.5-0.8B · 8da4w · XNNPACK") {

  @Volatile private var llm: LlmModule? = null

  override fun cancel() {
    llm?.stop()
  }

  override fun prepare(dir: File) {
    val model = File(dir, "model.pte")
    val tokenizer = File(dir, "tokenizer.json")
    if (!model.exists() || !tokenizer.exists()) return
    llm =
        LlmModule(
            LlmModuleConfig.create()
                .modulePath(model.absolutePath)
                .tokenizerPath(tokenizer.absolutePath)
                .temperature(0.7f)
                // The builder defaults dataPath to "" and the runner treats an empty string as a
                // real path, then aborts the process when it cannot map it. null is the fix.
                .dataPath(null)
                .modelType(LlmModuleConfig.MODEL_TYPE_TEXT)
                .build())
  }

  override fun run(dir: File, sink: TextSink) {
    // The question goes up first, so the screen reads as a prompt waiting for an answer.
    sink.body(PROMPT)
    val llm = this.llm
    if (llm == null) {
      sink.failed("push model.pte and tokenizer.json")
      return
    }
    try {
      sink.stats("thinking…")
      val reply = StringBuilder()
      // Qwen3.5 is a reasoning model: left alone it opens <think> and can spend thousands of
      // tokens there before answering. Pre-closing an empty think block is what the reference
      // template does for enable_thinking=false, and it is the difference between a two-second
      // answer and a seven-minute one.
      val turn = "<|im_start|>user\n$PROMPT<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
      llm.generate(
          turn,
          LlmGenerationConfig.create().maxNewTokens(64).temperature(0.7f).echo(false).build(),
          object : LlmCallback {
            override fun onResult(result: String) {
              // A short reply can arrive in one chunk that ends with the stop marker, so strip it
              // rather than dropping the chunk and losing the answer with it.
              reply.append(result.substringBefore("<|im_end|>"))
              // The model marks emphasis in Markdown, which on a plain TextView is just stray
              // asterisks in the middle of the answer.
              val answer = reply.toString().replace("*", "").trimStart()
              sink.body("$PROMPT\n\n$answer")
              sink.stats("generating…")
            }

            override fun onStats(stats: String) = sink.stats(summarize(stats))
          })
    } finally {
      sink.done()
    }
  }

  /** The runner reports JSON; the numbers a viewer cares about are prefill and decode rate. */
  private fun summarize(stats: String): String {
    val prompt = number(stats, "prompt_eval_end_ms") - number(stats, "inference_start_ms")
    val total = number(stats, "inference_end_ms") - number(stats, "inference_start_ms")
    val generated = number(stats, "generated_token_count")
    if (total <= 0.0 || generated <= 0.0) return "generating…"
    return "%.0f tok/s decode  ·  %.0f ms prefill".format(generated / ((total - prompt) / 1000.0), prompt)
  }

  private fun number(json: String, key: String): Double =
      Regex("\"$key\"\\s*:\\s*([0-9.]+)").find(json)?.groupValues?.get(1)?.toDoubleOrNull() ?: 0.0

  private companion object {
    const val PROMPT = "In two sentences: why run a model on the phone instead of a server?"
  }
}

/**
 * Whisper: a WAV on disk becomes text, one token at a time.
 *
 * `AsrModule` wants a single `.pte` exposing `encoder` and `text_decoder`, plus a separate
 * preprocessor `.pte` that turns the waveform into log-mel — so even the spectrogram runs through
 * ExecuTorch here, not through a hand-written FFT in Kotlin.
 */
class Listen : TextAct("Speech recognition", "Whisper · XNNPACK") {

  @Volatile private var asr: AsrModule? = null
  @Volatile private var wavPath: String? = null

  override fun prepare(dir: File) {
    val model = File(dir, "asr.pte")
    val tokenizer = File(dir, "asr_tokenizer.json")
    val preprocessor = File(dir, "asr_preprocess.pte")
    val wav = File(dir, "speech.wav")
    if (listOf(model, tokenizer, preprocessor, wav).any { !it.exists() }) return
    asr =
        AsrModule(
            modelPath = model.absolutePath,
            tokenizerPath = tokenizer.absolutePath,
            preprocessorPath = preprocessor.absolutePath,
        )
    wavPath = wav.absolutePath
  }

  override fun run(dir: File, sink: TextSink) {
    val asr = this.asr
    val wav = wavPath
    if (asr == null || wav == null) {
      sink.failed("push asr.pte, asr_preprocess.pte, asr_tokenizer.json, speech.wav")
      return
    }
    try {
      val transcript = StringBuilder()
      val start = System.nanoTime()
      asr.transcribe(
          wav,
          AsrTranscribeConfig(
              maxNewTokens = 96,
              temperature = 0f,
              decoderStartTokenId = START_OF_TRANSCRIPT,
          ),
          object : AsrCallback {
            override fun onToken(token: String) {
              transcript.append(token)
              sink.body(SPECIAL.replace(transcript, "").trim())
              sink.stats("%.1f s of audio · transcribing…".format(AUDIO_SECONDS))
            }

            override fun onError(errorCode: Int, message: String) =
                sink.failed("error $errorCode: $message")
          },
      )
      sink.stats("%.0f ms for %.1f s of speech".format((System.nanoTime() - start) / 1e6, AUDIO_SECONDS))
      android.util.Log.i("ETShowreel", "transcript: ${SPECIAL.replace(transcript, "").trim()}")
    } finally {
      sink.done()
    }
  }

  private companion object {
    /** Length of the bundled clip, used only for the caption. */
    const val AUDIO_SECONDS = 8.0

    /**
     * Whisper's `<|startoftranscript|>`. The runner takes the first decoder token from the config
     * and does not fall back to the model's own `decoder_start_token_id` method, so leaving this at
     * the default 0 starts decoding from `!` and the model answers with a confused mixture of
     * language and task tokens rather than a transcript.
     */
    const val START_OF_TRANSCRIPT = 50258L

    /** The language and task markers are part of the protocol, not part of what was said. */
    val SPECIAL = Regex("<\\|[^|]*\\|>")
  }
}

// ─── the reel ─────────────────────────────────────────────────────────────────

/**
 * The running order. Camera acts first so the reel opens on motion, then the two that hold still
 * and stream text.
 */
fun reel(): List<Act> = listOf(Detect(), Segment(), Depth(), Matte(), Chat(), Listen())
