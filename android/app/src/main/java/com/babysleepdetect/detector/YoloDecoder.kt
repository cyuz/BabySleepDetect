package com.babysleepdetect.detector

import com.babysleepdetect.posture.Posture
import com.babysleepdetect.posture.PostureClassifier
import com.babysleepdetect.posture.PostureConfig
import com.babysleepdetect.posture.PostureResult
import kotlin.math.max
import kotlin.math.min

/** One detected object, coordinates in model-input pixels until [Letterbox.toSource] is applied. */
data class Detection(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val score: Float,
    val cls: Int,
    val keypoints: FloatArray? = null, // 17 × (x, y, conf) for pose models
) {
    val area get() = max(0f, x2 - x1) * max(0f, y2 - y1)
}

/** Maps between the original frame and the square, padded model input. */
data class Letterbox(val scale: Float, val padX: Float, val padY: Float) {
    fun toSource(d: Detection): Detection {
        fun sx(v: Float) = (v - padX) / scale
        fun sy(v: Float) = (v - padY) / scale
        val k = d.keypoints?.copyOf()?.also {
            for (i in it.indices step 3) {
                it[i] = sx(it[i])
                it[i + 1] = sy(it[i + 1])
            }
        }
        return d.copy(x1 = sx(d.x1), y1 = sy(d.y1), x2 = sx(d.x2), y2 = sy(d.y2), keypoints = k)
    }

    companion object {
        fun of(srcW: Int, srcH: Int, size: Int): Letterbox {
            val scale = min(size.toFloat() / srcW, size.toFloat() / srcH)
            return Letterbox(scale, (size - srcW * scale) / 2f, (size - srcH * scale) / 2f)
        }
    }
}

enum class ModelTask { POSE, DETECT, CLASSIFY }

/** Decodes Ultralytics YOLO26 ONNX outputs. */
object YoloDecoder {
    private const val KPT_VALUES = PostureClassifier.NUM_KPTS * 3

    /** End-to-end (NMS-free) output, rows of [x1, y1, x2, y2, score, cls, kpts...]. */
    fun decodeEnd2End(rows: Array<FloatArray>, confThr: Float): List<Detection> =
        rows.filter { it[4] >= confThr }.map { r ->
            Detection(
                r[0], r[1], r[2], r[3], r[4], r[5].toInt(),
                if (r.size >= 6 + KPT_VALUES) r.copyOfRange(6, 6 + KPT_VALUES) else null,
            )
        }

    /**
     * Raw output (exported with end2end=False), channels-first [4 + nc (+ 51), N]
     * of [cx, cy, w, h, class scores..., kpts...]; needs NMS.
     */
    fun decodeRaw(out: Array<FloatArray>, numClasses: Int, confThr: Float, iouThr: Float = 0.45f): List<Detection> {
        val n = out[0].size
        val hasKpts = out.size >= 4 + numClasses + KPT_VALUES
        val dets = ArrayList<Detection>()
        for (j in 0 until n) {
            var best = 0
            for (c in 1 until numClasses) if (out[4 + c][j] > out[4 + best][j]) best = c
            val score = out[4 + best][j]
            if (score < confThr) continue
            val cx = out[0][j]
            val cy = out[1][j]
            val w = out[2][j]
            val h = out[3][j]
            val kpts = if (hasKpts) FloatArray(KPT_VALUES) { out[4 + numClasses + it][j] } else null
            dets += Detection(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score, best, kpts)
        }
        return nms(dets, iouThr)
    }

    fun nms(dets: List<Detection>, iouThr: Float): List<Detection> {
        val kept = ArrayList<Detection>()
        for (d in dets.sortedByDescending { it.score }) {
            if (kept.none { it.cls == d.cls && iou(it, d) > iouThr }) kept += d
        }
        return kept
    }

    fun iou(a: Detection, b: Detection): Float {
        val iw = max(0f, min(a.x2, b.x2) - max(a.x1, b.x1))
        val ih = max(0f, min(a.y2, b.y2) - max(a.y1, b.y1))
        val inter = iw * ih
        val union = a.area + b.area - inter
        return if (union <= 0f) 0f else inter / union
    }

    /** Parses Ultralytics' `names` metadata, e.g. "{0: 'supine', 1: 'prone'}". */
    fun parseNames(meta: String?): Map<Int, String> {
        if (meta == null) return emptyMap()
        return Regex("""(\d+)\s*:\s*['"]([^'"]*)['"]""").findAll(meta)
            .associate { it.groupValues[1].toInt() to it.groupValues[2] }
    }

    fun postureFromClassName(name: String): Posture {
        val n = name.lowercase()
        return when {
            "prone" in n || "stomach" in n -> Posture.PRONE
            "supine" in n || "back" in n -> Posture.SUPINE
            "side" in n -> Posture.SIDE
            else -> Posture.UNKNOWN
        }
    }

    /** Picks the baby (largest subject in a crib view) and turns it into a posture. */
    fun interpret(
        task: ModelTask,
        dets: List<Detection>,
        names: Map<Int, String>,
        cfg: PostureConfig,
    ): Pair<PostureResult, Detection?> {
        val baby = dets.maxByOrNull { it.area }
            ?: return PostureResult(Posture.UNKNOWN, 0f, "no baby detected") to null
        val result = when {
            task == ModelTask.POSE && baby.keypoints != null -> PostureClassifier.classify(baby.keypoints, cfg)
            else -> {
                val name = names[baby.cls] ?: baby.cls.toString()
                PostureResult(postureFromClassName(name), baby.score, "class '$name'")
            }
        }
        return result to baby
    }

    /** Top-1 posture from a classification model's probabilities. */
    fun interpretClassify(probs: FloatArray, names: Map<Int, String>): PostureResult {
        val top = probs.indices.maxByOrNull { probs[it] } ?: return PostureResult(Posture.UNKNOWN, 0f, "empty output")
        val name = names[top] ?: top.toString()
        return PostureResult(postureFromClassName(name), probs[top], "class '$name'")
    }
}
