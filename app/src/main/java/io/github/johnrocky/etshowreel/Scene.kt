package io.github.johnrocky.etshowreel

import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import java.io.File

/**
 * A clip standing in for the camera.
 *
 * The reel normally runs on live frames, but a recording made that way shows whatever the lens
 * happened to be pointed at. Dropping a `scene.mp4` next to the models makes the run reproducible:
 * the same frames, the same detections, every time. If the file is absent the app uses the camera
 * and this class is never constructed.
 *
 * Frames come out by wall-clock position rather than one per inference, so the scene plays at its
 * own speed no matter how long a model takes, and loops when it runs out.
 */
class Scene(file: File) : AutoCloseable {

  private val retriever =
      MediaMetadataRetriever().apply { setDataSource(file.absolutePath) }

  /** Clip length in microseconds; 0 if the file did not declare one. */
  private val durationUs: Long =
      (retriever
              .extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
              ?.toLongOrNull() ?: 0L) * 1000L

  val isUsable: Boolean
    get() = durationUs > 0

  /**
   * The frame [elapsedUs] into the clip, counted from [fromUs] and wrapping at the end.
   *
   * The caller decides where in the clip to start rather than the clip playing on one clock, so an
   * act can be given the stretch of footage that suits it: a street for the detector, a portrait
   * for the matting model. Otherwise which act sees which scene is down to how the reel's cycle
   * happens to line up with the clip's length.
   *
   * `OPTION_CLOSEST` decodes to the exact position rather than snapping to the nearest keyframe,
   * which for a short clip is the difference between motion and a slideshow.
   */
  fun frame(fromUs: Long, elapsedUs: Long): Bitmap? {
    if (durationUs <= 0) return null
    return retriever.getFrameAtTime(
        (fromUs + elapsedUs) % durationUs,
        MediaMetadataRetriever.OPTION_CLOSEST,
    )
  }

  override fun close() = retriever.release()
}
