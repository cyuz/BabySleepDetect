package com.babysleepdetect.posture

import kotlin.math.hypot
import kotlin.math.max

/**
 * Sleep-posture rules from COCO-17 keypoints of a YOLO26 pose model.
 * Kotlin port of babysleep/posture.py; keep the two in sync.
 */
enum class Posture(val zh: String) {
    SUPINE("仰睡"),
    PRONE("趴睡"),
    SIDE("側睡"),
    UNKNOWN("未知"),
}

data class PostureResult(val posture: Posture, val confidence: Float, val reason: String)

data class PostureConfig(
    val kptConf: Float = 0.5f, // a keypoint counts as visible above this confidence
    val sideRatio: Float = 0.35f, // shoulder width / torso length below this => side-lying
)

object PostureClassifier {
    const val NUM_KPTS = 17
    private const val NOSE = 0
    private const val L_EYE = 1
    private const val R_EYE = 2
    private const val L_SHOULDER = 5
    private const val R_SHOULDER = 6
    private const val L_HIP = 11
    private const val R_HIP = 12

    /** Classify one person's posture from a flat array of 17 × (x, y, conf). */
    fun classify(kpts: FloatArray, cfg: PostureConfig = PostureConfig()): PostureResult {
        require(kpts.size == NUM_KPTS * 3) { "expected ${NUM_KPTS * 3} values, got ${kpts.size}" }
        val thr = cfg.kptConf
        fun x(i: Int) = kpts[i * 3]
        fun y(i: Int) = kpts[i * 3 + 1]
        fun c(i: Int) = kpts[i * 3 + 2]
        fun visible(i: Int) = c(i) >= thr
        fun mid(a: Int, b: Int): FloatArray? {
            val pts = listOf(a, b).filter { visible(it) }
            if (pts.isEmpty()) return null
            return floatArrayOf(pts.map { x(it) }.average().toFloat(), pts.map { y(it) }.average().toFloat())
        }

        val faceConf = (c(NOSE) + c(L_EYE) + c(R_EYE)) / 3f
        val faceVisible = faceConf >= thr
        val bothShoulders = visible(L_SHOULDER) && visible(R_SHOULDER)
        val shoulderMid = mid(L_SHOULDER, R_SHOULDER)
        val hipMid = mid(L_HIP, R_HIP)
        val torsoVisible = shoulderMid != null || hipMid != null

        if (!torsoVisible && !faceVisible) return PostureResult(Posture.UNKNOWN, 0f, "body not visible")

        // Facing direction from the body's own frame: with "up" = hips -> shoulders,
        // a body facing the camera has its left shoulder on the right of that axis.
        var facing = 0f // >0 facing camera, <0 back to camera
        if (bothShoulders && hipMid != null && shoulderMid != null) {
            val upX = shoulderMid[0] - hipMid[0]
            val upY = shoulderMid[1] - hipMid[1]
            val lrX = x(L_SHOULDER) - x(R_SHOULDER)
            val lrY = y(L_SHOULDER) - y(R_SHOULDER)
            val torsoLen = hypot(upX, upY)
            val shoulderW = hypot(lrX, lrY)
            if (torsoLen > 1e-6f) {
                if (shoulderW / torsoLen < cfg.sideRatio) {
                    return PostureResult(Posture.SIDE, 0.6f, "narrow shoulders relative to torso")
                }
                val cross = upX * lrY - upY * lrX
                facing = (cross / (torsoLen * max(shoulderW, 1e-6f))).coerceIn(-1f, 1f)
            }
        }

        if (faceVisible) {
            if (facing < -0.5f) {
                // Face keypoints seen while the torso faces away: head turned while prone.
                return PostureResult(Posture.PRONE, 0.6f, "torso facing away, head turned")
            }
            return PostureResult(Posture.SUPINE, faceConf, "face visible")
        }

        // Face not visible but body is: face-down or face covered — both are unsafe.
        if (facing < -0.3f) return PostureResult(Posture.PRONE, 0.9f, "face hidden, back facing camera")
        return PostureResult(Posture.PRONE, 0.7f, "face hidden or covered")
    }
}
