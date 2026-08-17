package io.github.johnrocky.etshowreel

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import io.github.johnrocky.etvision.android.rgbaToBitmap
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.extension.image.ImageOrientation
import org.pytorch.executorch.extension.image.ImageProcessor
import org.pytorch.executorch.extension.image.ImageProcessorConfig
import org.pytorch.executorch.extension.image.Normalization
import org.pytorch.executorch.extension.image.YuvFormat

private const val TAG = "ETShowreel"
private const val ACCENT = 0xFF6FE3B4.toInt()

/** How long a camera act holds the screen. Text acts run until the model stops. */
private const val VISION_SECONDS = 7

/** A text act that never finishes still has to hand the screen back. */
private const val TEXT_TIMEOUT_SECONDS = 40

/**
 * A full-screen reel: one model at a time, filling the frame, advancing on its own.
 *
 * Camera acts show the model's output over the live frame; text acts take the screen over and
 * stream tokens. Every act carries its own measured latency on screen, so the recording is its own
 * evidence.
 *
 * The camera path is the reason ExecuTorch's `ImageProcessor` matters here: CameraX hands over YUV
 * planes and `processYuv` takes them directly, rather than making the app convert to a Bitmap first
 * and pay for it on every frame.
 */
class MainActivity : AppCompatActivity() {

  private lateinit var output: ImageView
  private lateinit var scrim: android.view.View
  private lateinit var bodyView: TextView
  private lateinit var titleView: TextView
  private lateinit var subtitleView: TextView
  private lateinit var statsView: TextView

  private val acts = reel()
  private val clock = android.os.Handler(android.os.Looper.getMainLooper())
  private val busy = AtomicBoolean(false)
  private val worker = Executors.newSingleThreadExecutor()

  @Volatile private var actIndex = 0
  /** Guards against a finished text act and the timeout both advancing the reel. */
  @Volatile private var advanced = false

  // Owned by [worker]: loading and inference happen on one thread so a swap can never land while
  // forward() is running, which the runtime refuses outright.
  private val modules = mutableMapOf<VisionAct, Module>()
  private var module: Module? = null
  private var processor: ImageProcessor? = null
  private var scene: Scene? = null

  /** The most recent source frame with nothing drawn on it, for the moment an act changes. */
  @Volatile private var plainFrame: Bitmap? = null

  /** Held so the warm-up buffers outlive the inference that still points at them. */
  private val warmupInputs = mutableListOf<org.pytorch.executorch.Tensor>()

  private val act: Act
    get() = acts[actIndex]

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    buildUi()
    if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) !=
        PackageManager.PERMISSION_GRANTED) {
      ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1)
    } else {
      startFrames()
    }
  }

  override fun onRequestPermissionsResult(
      requestCode: Int,
      permissions: Array<out String>,
      grantResults: IntArray,
  ) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) startFrames()
  }

  override fun onDestroy() {
    super.onDestroy()
    worker.shutdown()
    modules.values.forEach { it.destroy() }
    processor?.close()
    scene?.close()
  }

  /**
   * Starts whichever frame source is available: a `scene.mp4` dropped next to the models if there
   * is one, the camera otherwise. A clip makes a recording reproducible; the camera is the point of
   * the app.
   */
  private fun startFrames() {
    val dir = getExternalFilesDir(null)
    val clip = dir?.let { File(it, "scene.mp4") }
    if (clip != null && clip.exists()) {
      val loaded = Scene(clip)
      if (loaded.isUsable) {
        scene = loaded
        Log.i(TAG, "playing ${clip.name} in place of the camera")
        preload()
        enterAct(0)
        worker.execute { playScene() }
        return
      }
      loaded.close()
      Log.w(TAG, "${clip.name} declared no duration; falling back to the camera")
    }
    startCamera()
  }

  private fun startCamera() {
    val future = ProcessCameraProvider.getInstance(this)
    future.addListener(
        {
          val provider = future.get()
          val analysis =
              ImageAnalysis.Builder()
                  .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                  .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                  .build()
          analysis.setAnalyzer(worker) { proxy -> onFrame(proxy) }
          provider.unbindAll()
          provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
          preload()
          enterAct(0)
        },
        mainExecutor,
    )
  }

  /**
   * Feeds the vision acts from the clip, one frame per inference, re-queuing itself so the work
   * stays on the same thread that owns the modules.
   */
  private fun playScene() {
    val source = scene ?: return
    val current = act
    val model = module
    val processor = this.processor
    if (current is TextAct) {
      // Text acts have nothing of their own to show, and a black screen for the length of a
      // generation is dead air. The clip keeps playing behind the words, dimmed by the scrim.
      val bitmap = source.frame()
      if (bitmap != null) {
        val plain = bitmap.copy(Bitmap.Config.ARGB_8888, false)
        bitmap.recycle()
        plainFrame = plain
        runOnUiThread { if (act === current) output.setImageBitmap(plain) }
      }
    } else if (current is VisionAct && model != null && processor != null) {
      try {
        val bitmap = source.frame()
        if (bitmap != null) {
          val upright = bitmap.copy(Bitmap.Config.ARGB_8888, false)
          bitmap.recycle()
          plainFrame = upright
          val input = processor.process(upright)
          val runStart = System.nanoTime()
          val outputs = model.forward(EValue.from(input))
          val runMs = (System.nanoTime() - runStart) / 1e6
          val frame = current.render(outputs, upright, current.mapperFor(upright))
          runOnUiThread {
            if (act === current) {
              output.setImageBitmap(frame.bitmap)
              statsView.text = "%.0f ms  ·  %s".format(runMs, frame.detail)
            }
          }
        }
      } catch (e: Throwable) {
        Log.e(TAG, "scene frame failed on ${current.title}", e)
      }
    }
    if (!worker.isShutdown) worker.execute { playScene() }
  }

  // ─── reel ───────────────────────────────────────────────────────────────────

  /**
   * Opens every model up front rather than when its act arrives.
   *
   * Loading is the long pole here, not inference: a vision model takes about three seconds to map
   * and initialize against a seven-second act, and the LLM's 620 MB takes longer still. Loading on
   * arrival spent most of the reel showing "loading". They all stay open for the session instead.
   *
   * The vision models go on [worker] because that thread owns them; the two text models are opened
   * on their own thread so the reel's first frames are not stuck behind 850 MB of weights.
   */
  private fun preload() {
    val dir = getExternalFilesDir(null) ?: return
    for (a in acts.filterIsInstance<VisionAct>()) {
      worker.execute {
        val start = System.nanoTime()
        val m = a.load(dir)
        if (m == null) {
          Log.w(TAG, "${a.file} not pushed")
        } else {
          // Opening a .pte is an mmap and costs almost nothing; the delay is in the first
          // forward(), where XNNPACK builds its plan. Paying that here with a zero frame is what
          // actually removes the pause at the start of each act.
          m.forward(EValue.from(blankInput(a)))
          modules[a] = m
          if (act === a) module = m
          Log.i(TAG, "warmed ${a.file} in %.0f ms".format((System.nanoTime() - start) / 1e6))
        }
      }
    }
    thread(name = "preload-text") {
      for (a in acts.filterIsInstance<TextAct>()) {
        val start = System.nanoTime()
        try {
          a.prepare(dir)
          Log.i(TAG, "opened ${a.title} in %.0f ms".format((System.nanoTime() - start) / 1e6))
        } catch (e: Throwable) {
          Log.e(TAG, "preload failed for ${a.title}", e)
        }
      }
    }
  }

  private fun enterAct(index: Int) {
    actIndex = index
    advanced = false

    val current = act
    titleView.text = current.title
    subtitleView.text = current.subtitle
    statsView.text = ""
    scrim.visibility = if (current is TextAct) android.view.View.VISIBLE else android.view.View.GONE
    bodyView.visibility = if (current is TextAct) android.view.View.VISIBLE else android.view.View.GONE
    if (current is TextAct) bodyView.text = ""
    Log.i(TAG, "act ${current.title}")

    // Until the incoming model produces something, the screen would otherwise keep the outgoing
    // model's overlay under the new act's title, which reads as the reel showing the wrong answer.
    // Until the incoming model produces something, the screen would otherwise keep the outgoing
    // model's overlay under the new act's title, which reads as the reel showing the wrong answer.
    // The last bare frame is already in hand, so the swap costs nothing.
    if (current is VisionAct) {
      plainFrame?.let { output.setImageBitmap(it) }
      statsView.text = "starting…"
    }

    // The swap is queued behind whatever frame is in flight; doing it here would tear the module
    // out from under a running forward().
    worker.execute { swapModel(current) }
    if (current is TextAct) enterText(current)

    // The deadline is its own timer rather than something the frame loop checks, because a text
    // act leaves the frame loop with nothing to do and the reel would sit there forever.
    clock.removeCallbacksAndMessages(null)
    val hold = if (current is VisionAct) VISION_SECONDS else TEXT_TIMEOUT_SECONDS
    clock.postDelayed({ advance() }, hold * 1000L)
  }

  /**
   * A zero frame shaped the way an act's model expects, used only to force the first plan build.
   *
   * The buffer is direct and kept for the life of the activity on purpose: a method holds a pointer
   * to whatever it was last given, so a warm-up input allocated on the Java heap is a pointer into
   * memory the collector is free to move, and the next real inference faults on it.
   */
  private fun blankInput(a: VisionAct): org.pytorch.executorch.Tensor {
    val numel = 3 * a.config.targetHeight * a.config.targetWidth
    val tensor =
        org.pytorch.executorch.Tensor.fromBlob(
            org.pytorch.executorch.Tensor.allocateFloatBuffer(numel),
            longArrayOf(1, 3, a.config.targetHeight.toLong(), a.config.targetWidth.toLong()),
        )
    warmupInputs += tensor
    return tensor
  }

  private fun swapModel(current: Act) {
    processor?.close()
    processor = null
    if (current !is VisionAct) {
      module = null
      return
    }
    module = modules[current]
    processor = ImageProcessor(current.config)
    val loaded = module != null
    runOnUiThread {
      if (act === current) statsView.text = if (loaded) "running…" else "${current.file} not pushed"
    }
  }

  private fun enterText(current: TextAct) {
    val dir = getExternalFilesDir(null) ?: return
    val forAct = actIndex
    val sink =
        object : TextSink {
          private fun post(block: () -> Unit) = runOnUiThread { if (actIndex == forAct) block() }

          override fun body(text: String) = post { bodyView.text = text }

          override fun stats(line: String) = post { statsView.text = line }

          override fun failed(reason: String) = post {
            bodyView.text = reason
            statsView.text = "unavailable"
          }

          override fun done() = post { advance() }
        }
    thread(name = "act-${current.title}") {
      // Below DISPLAY the scheduler parks generation on the little cores and the token rate on
      // screen stops being the number the device is actually capable of.
      android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_DISPLAY)
      try {
        current.run(dir, sink)
      } catch (e: Throwable) {
        Log.e(TAG, "act ${current.title} failed", e)
        sink.failed(e.message ?: e.javaClass.simpleName)
        sink.done()
      }
    }
  }

  /** Advances once per act, whoever gets there first — the act finishing or its clock running out. */
  private fun advance() {
    if (advanced) return
    advanced = true
    clock.removeCallbacksAndMessages(null)
    enterAct((actIndex + 1) % acts.size)
  }

  // ─── camera ─────────────────────────────────────────────────────────────────

  private fun onFrame(proxy: ImageProxy) {
    val current = act
    val model = module
    val processor = this.processor
    if (current !is VisionAct || model == null || processor == null ||
        !busy.compareAndSet(false, true)) {
      proxy.close()
      return
    }
    try {
      val yPlane = proxy.planes[0]
      // planes[1] and planes[2] are views one byte apart into the same interleaved allocation;
      // planes[1] read as NV12 is the one whose first byte is U.
      val uvPlane = proxy.planes[1]
      if (uvPlane.pixelStride != 2) {
        // Fully planar I420 cannot be read as semi-planar; skip rather than show garbage.
        return
      }
      // CameraX reports the rotation needed to display the frame upright, which is the same
      // convention the processor's EXIF orientation uses.
      val orientation =
          when (proxy.imageInfo.rotationDegrees) {
            90 -> ImageOrientation.RIGHT
            180 -> ImageOrientation.DOWN
            270 -> ImageOrientation.LEFT
            else -> ImageOrientation.UP
          }

      val input =
          processor.processYuv(
              yPlane.buffer,
              yPlane.rowStride,
              uvPlane.buffer,
              uvPlane.rowStride,
              proxy.width,
              proxy.height,
              YuvFormat.NV12,
              orientation,
          )

      val runStart = System.nanoTime()
      val outputs = model.forward(EValue.from(input))
      val runMs = (System.nanoTime() - runStart) / 1e6

      // The overlay has to be drawn on the oriented frame, not the raw one.
      val source = uprightBitmap(yPlane, uvPlane, proxy, orientation)
      val frame = current.render(outputs, source, current.mapperFor(source))

      runOnUiThread {
        if (act === current) {
          output.setImageBitmap(frame.bitmap)
          statsView.text = "%.0f ms  ·  %s".format(runMs, frame.detail)
        }
      }
    } catch (e: Throwable) {
      Log.e(TAG, "frame failed on ${current.title}", e)
    } finally {
      busy.set(false)
      proxy.close()
    }
  }

  /**
   * Re-runs the processor at the frame's own size to get an upright RGB copy to draw on. Cheaper
   * than a second conversion in Kotlin, and it reuses the orientation handling that already ran.
   */
  private fun uprightBitmap(
      yPlane: ImageProxy.PlaneProxy,
      uvPlane: ImageProxy.PlaneProxy,
      proxy: ImageProxy,
      orientation: ImageOrientation,
  ): Bitmap {
    val rotated = orientation == ImageOrientation.RIGHT || orientation == ImageOrientation.LEFT
    val w = if (rotated) proxy.height else proxy.width
    val h = if (rotated) proxy.width else proxy.height
    val display =
        ImageProcessor(
            ImageProcessorConfig(
                targetWidth = w,
                targetHeight = h,
                normalization = Normalization.zeroToOne(),
            ))
    try {
      val chw =
          display
              .processYuv(
                  yPlane.buffer,
                  yPlane.rowStride,
                  uvPlane.buffer,
                  uvPlane.rowStride,
                  proxy.width,
                  proxy.height,
                  YuvFormat.NV12,
                  orientation,
              )
              .dataAsFloatArray
      val plane = w * h
      val rgba = ByteArray(plane * 4)
      for (i in 0 until plane) {
        rgba[i * 4] = ((chw[i] * 255f).toInt().coerceIn(0, 255)).toByte()
        rgba[i * 4 + 1] = ((chw[plane + i] * 255f).toInt().coerceIn(0, 255)).toByte()
        rgba[i * 4 + 2] = ((chw[2 * plane + i] * 255f).toInt().coerceIn(0, 255)).toByte()
        rgba[i * 4 + 3] = 255.toByte()
      }
      return rgbaToBitmap(rgba, w, h)
    } finally {
      display.close()
    }
  }

  // ─── ui ─────────────────────────────────────────────────────────────────────

  private fun buildUi() {
    val root = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }
    val fill =
        FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT,
        )

    output =
        ImageView(this).apply {
          scaleType = ImageView.ScaleType.CENTER_CROP
          layoutParams = fill
        }
    root.addView(output)

    scrim =
        android.view.View(this).apply {
          setBackgroundColor(0xD0000000.toInt())
          visibility = android.view.View.GONE
          layoutParams = fill
        }

    // Text acts get the same full frame the camera acts get. Bottom gravity means the first
    // tokens land just above the caption and the block grows upward, which reads better than
    // text crawling down from the top of an empty screen. The bottom padding has to clear the
    // caption band, which is drawn over this view -- without it the first lines are invisible.
    bodyView =
        TextView(this).apply {
          setTextColor(Color.WHITE)
          textSize = 26f
          setLineSpacing(0f, 1.25f)
          gravity = Gravity.BOTTOM
          setPadding(dp(24), dp(72), dp(24), dp(200))
          visibility = android.view.View.GONE
          layoutParams = fill
        }
    root.addView(scrim)
    root.addView(bodyView)

    val caption =
        LinearLayout(this).apply {
          orientation = LinearLayout.VERTICAL
          setPadding(dp(20), dp(0), dp(20), dp(36))
          setBackgroundColor(0xC0000000.toInt())
          layoutParams =
              FrameLayout.LayoutParams(
                      ViewGroup.LayoutParams.MATCH_PARENT,
                      ViewGroup.LayoutParams.WRAP_CONTENT,
                  )
                  .apply { gravity = Gravity.BOTTOM }
        }
    titleView =
        TextView(this).apply {
          setTextColor(Color.WHITE)
          textSize = 30f
          setTypeface(typeface, Typeface.BOLD)
          setPadding(0, dp(16), 0, 0)
        }
    subtitleView =
        TextView(this).apply {
          setTextColor(0xFFBBBBBB.toInt())
          textSize = 16f
        }
    statsView =
        TextView(this).apply {
          setTextColor(ACCENT)
          textSize = 22f
          setTypeface(typeface, Typeface.BOLD)
          setPadding(0, dp(6), 0, 0)
        }
    caption.addView(titleView)
    caption.addView(subtitleView)
    caption.addView(statsView)
    root.addView(caption)
    setContentView(root)
  }

  private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()
}
