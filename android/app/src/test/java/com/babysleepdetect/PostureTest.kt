package com.babysleepdetect

import com.babysleepdetect.detector.Detection
import com.babysleepdetect.detector.Letterbox
import com.babysleepdetect.detector.ModelTask
import com.babysleepdetect.detector.YoloDecoder
import com.babysleepdetect.posture.MonitorConfig
import com.babysleepdetect.posture.Posture
import com.babysleepdetect.posture.PostureClassifier
import com.babysleepdetect.posture.PostureConfig
import com.babysleepdetect.posture.SleepMonitor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin

/** Mirrors tests/test_posture.py so the Kotlin port stays in sync with the Python rules. */
class PostureTest {
    private fun kpts(
        face: Boolean = true,
        facingCamera: Boolean = true,
        shoulderW: Float = 60f,
        rotateDeg: Double = 0.0,
    ): FloatArray {
        val k = FloatArray(51)
        fun set(i: Int, x: Float, y: Float, c: Float) {
            k[i * 3] = x; k[i * 3 + 1] = y; k[i * 3 + 2] = c
        }
        val half = shoulderW / 2
        // From the camera, a body facing it has its left shoulder on the image right.
        val s = if (facingCamera) 1 else -1
        set(5, 100 + s * half, 100f, 0.9f)
        set(6, 100 - s * half, 100f, 0.9f)
        set(11, 100f + s * 20, 200f, 0.9f)
        set(12, 100f - s * 20, 200f, 0.9f)
        set(3, 100f + s * 15, 60f, 0.8f)
        set(4, 100f - s * 15, 60f, 0.8f)
        if (face) {
            set(0, 100f, 55f, 0.9f)
            set(1, 110f, 45f, 0.9f)
            set(2, 90f, 45f, 0.9f)
        }
        if (rotateDeg != 0.0) {
            val a = Math.toRadians(rotateDeg)
            for (i in 0 until 17) {
                val x = k[i * 3] - 100
                val y = k[i * 3 + 1] - 100
                k[i * 3] = (x * cos(a) - y * sin(a)).toFloat() + 100
                k[i * 3 + 1] = (x * sin(a) + y * cos(a)).toFloat() + 100
            }
        }
        return k
    }

    private val rotations = listOf(0.0, 45.0, 90.0, 180.0, 270.0)

    @Test fun supineAnyOrientation() = rotations.forEach {
        assertEquals("rot $it", Posture.SUPINE, PostureClassifier.classify(kpts(rotateDeg = it)).posture)
    }

    @Test fun proneBackToCamera() = rotations.forEach {
        val r = PostureClassifier.classify(kpts(face = false, facingCamera = false, rotateDeg = it))
        assertEquals("rot $it", Posture.PRONE, r.posture)
        assertTrue(r.confidence >= 0.9f)
    }

    @Test fun proneHeadTurned() =
        assertEquals(Posture.PRONE, PostureClassifier.classify(kpts(facingCamera = false)).posture)

    @Test fun faceCoveredIsProne() =
        assertEquals(Posture.PRONE, PostureClassifier.classify(kpts(face = false)).posture)

    @Test fun sideLying() =
        assertEquals(Posture.SIDE, PostureClassifier.classify(kpts(shoulderW = 10f)).posture)

    @Test fun nothingVisible() =
        assertEquals(Posture.UNKNOWN, PostureClassifier.classify(FloatArray(51)).posture)

    @Test fun monitorAlertsAfterPersistentProne() {
        val m = SleepMonitor(MonitorConfig(windowSeconds = 2.0, alertAfterSeconds = 5.0))
        var t = 0.0
        repeat(10) { assertFalse(m.update(Posture.SUPINE, t).alert); t += 0.1 }
        val states = (0 until 80).map { m.update(Posture.PRONE, t).also { t += 0.1 } }
        assertFalse(states[30].alert)
        assertTrue(states.last().alert)
        assertEquals(Posture.PRONE, states.last().posture)
    }

    @Test fun monitorIgnoresFlicker() {
        val m = SleepMonitor(MonitorConfig(windowSeconds = 2.0, alertAfterSeconds = 5.0))
        var t = 0.0
        for (i in 0 until 200) {
            assertFalse(m.update(if (i % 5 == 0) Posture.PRONE else Posture.SUPINE, t).alert)
            t += 0.1
        }
    }

    @Test fun monitorMissingBaby() {
        val m = SleepMonitor(MonitorConfig(missingAfterSeconds = 10.0))
        m.update(Posture.SUPINE, 0.0)
        assertFalse(m.update(Posture.UNKNOWN, 5.0).missing)
        assertTrue(m.update(Posture.UNKNOWN, 11.0).missing)
    }
}

class YoloDecoderTest {
    @Test fun parsesUltralyticsNames() {
        assertEquals(
            mapOf(0 to "supine", 1 to "prone", 2 to "side"),
            YoloDecoder.parseNames("{0: 'supine', 1: 'prone', 2: 'side'}"),
        )
        assertEquals(emptyMap<Int, String>(), YoloDecoder.parseNames(null))
    }

    @Test fun classNameMapping() {
        assertEquals(Posture.PRONE, YoloDecoder.postureFromClassName("baby_prone"))
        assertEquals(Posture.SUPINE, YoloDecoder.postureFromClassName("Supine"))
        assertEquals(Posture.SIDE, YoloDecoder.postureFromClassName("side"))
        assertEquals(Posture.UNKNOWN, YoloDecoder.postureFromClassName("person"))
    }

    @Test fun decodesEnd2EndPoseRows() {
        val row = FloatArray(57).also {
            it[0] = 10f; it[1] = 20f; it[2] = 110f; it[3] = 220f; it[4] = 0.9f; it[5] = 0f
            it[6] = 50f; it[7] = 30f; it[8] = 0.95f
        }
        val low = FloatArray(57).also { it[4] = 0.1f }
        val dets = YoloDecoder.decodeEnd2End(arrayOf(row, low), 0.25f)
        assertEquals(1, dets.size)
        assertEquals(51, dets[0].keypoints!!.size)
        assertEquals(50f, dets[0].keypoints!![0])
    }

    @Test fun decodesRawWithNms() {
        // Channels-first [4 + 1 class, 3 anchors]: two overlapping boxes and one separate box.
        val out = arrayOf(
            floatArrayOf(50f, 52f, 300f), // cx
            floatArrayOf(50f, 50f, 300f), // cy
            floatArrayOf(40f, 40f, 20f), // w
            floatArrayOf(40f, 40f, 20f), // h
            floatArrayOf(0.9f, 0.8f, 0.7f), // score
        )
        val dets = YoloDecoder.decodeRaw(out, numClasses = 1, confThr = 0.25f)
        assertEquals(2, dets.size)
        assertEquals(0.9f, dets[0].score)
        assertNull(dets[0].keypoints)
    }

    @Test fun letterboxMapsBackToSource() {
        val lb = Letterbox.of(1280, 720, 640) // scale 0.5, pad y 140
        val d = lb.toSource(Detection(0f, 140f, 640f, 500f, 1f, 0, floatArrayOf(320f, 320f, 1f)))
        assertEquals(0f, d.x1, 1e-3f)
        assertEquals(0f, d.y1, 1e-3f)
        assertEquals(1280f, d.x2, 1e-3f)
        assertEquals(720f, d.y2, 1e-3f)
        assertEquals(640f, d.keypoints!![0], 1e-3f)
        assertEquals(360f, d.keypoints!![1], 1e-3f)
    }

    @Test fun interpretPicksLargestDetection() {
        val small = Detection(0f, 0f, 10f, 10f, 0.9f, 1)
        val big = Detection(0f, 0f, 100f, 100f, 0.6f, 0)
        val names = mapOf(0 to "supine", 1 to "prone")
        val (result, baby) = YoloDecoder.interpret(ModelTask.DETECT, listOf(small, big), names, PostureConfig())
        assertEquals(Posture.SUPINE, result.posture)
        assertEquals(big, baby)
        assertEquals(Posture.UNKNOWN, YoloDecoder.interpret(ModelTask.POSE, emptyList(), names, PostureConfig()).first.posture)
    }

    @Test fun interpretClassify() {
        val r = YoloDecoder.interpretClassify(floatArrayOf(0.1f, 0.8f, 0.1f), mapOf(0 to "supine", 1 to "prone", 2 to "side"))
        assertEquals(Posture.PRONE, r.posture)
    }
}
