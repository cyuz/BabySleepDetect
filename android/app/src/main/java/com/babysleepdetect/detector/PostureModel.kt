package com.babysleepdetect.detector

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import com.babysleepdetect.posture.PostureConfig
import com.babysleepdetect.posture.PostureResult
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class FrameResult(
    val result: PostureResult,
    val baby: Detection?, // in source-bitmap pixels
    val inferenceMs: Long,
)

/**
 * Runs a YOLO26 ONNX model exported by scripts/export_android_model.py.
 * Supports the pose model (default) and custom detect / classify models from train.py.
 */
class PostureModel(context: Context, assetName: String = MODEL_ASSET) : Closeable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    val inputSize: Int
    val task: ModelTask
    private val names: Map<Int, String>
    private val end2end: Boolean

    private val input: ByteBuffer
    private val pixels: IntArray
    private val canvasBitmap: Bitmap
    private val canvas: Canvas
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    init {
        val bytes = context.assets.open(assetName).use { it.readBytes() }
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(1, 4))
        }
        session = env.createSession(bytes, opts)
        inputName = session.inputNames.first()
        inputSize = (session.inputInfo[inputName]!!.info as TensorInfo).shape.last().toInt()
        val meta = session.metadata.customMetadata
        task = when (meta["task"]) {
            "detect" -> ModelTask.DETECT
            "classify" -> ModelTask.CLASSIFY
            else -> ModelTask.POSE
        }
        names = YoloDecoder.parseNames(meta["names"])
        end2end = meta["end2end"]?.equals("true", ignoreCase = true) ?: false

        input = ByteBuffer.allocateDirect(3 * inputSize * inputSize * 4).order(ByteOrder.nativeOrder())
        pixels = IntArray(inputSize * inputSize)
        canvasBitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        canvas = Canvas(canvasBitmap)
    }

    fun run(frame: Bitmap, conf: Float, postureCfg: PostureConfig): FrameResult {
        val start = System.nanoTime()
        val lb = preprocess(frame)
        val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        val (result, baby) = OnnxTensor.createTensor(env, input.asFloatBuffer(), shape).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { out ->
                @Suppress("UNCHECKED_CAST")
                when (val value = out[0].value) {
                    is Array<*> -> when (val first = value[0]) {
                        is FloatArray -> YoloDecoder.interpretClassify(first, names) to null // [1, C]
                        is Array<*> -> {
                            val m = first as Array<FloatArray> // [1, rows, cols]
                            val dets = if (end2end) {
                                YoloDecoder.decodeEnd2End(m, conf)
                            } else {
                                YoloDecoder.decodeRaw(m, names.size.coerceAtLeast(1), conf)
                            }
                            YoloDecoder.interpret(task, dets, names, postureCfg)
                        }
                        else -> error("unexpected output type")
                    }
                    else -> error("unexpected output type")
                }
            }
        }
        return FrameResult(result, baby?.let(lb::toSource), (System.nanoTime() - start) / 1_000_000)
    }

    /** Letterboxes [frame] into the gray square input and writes normalized CHW RGB floats. */
    private fun preprocess(frame: Bitmap): Letterbox {
        val lb = Letterbox.of(frame.width, frame.height, inputSize)
        canvas.drawColor(Color.rgb(114, 114, 114))
        val dst = RectF(lb.padX, lb.padY, inputSize - lb.padX, inputSize - lb.padY)
        canvas.drawBitmap(frame, null, dst, paint)
        canvasBitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        val plane = inputSize * inputSize
        val floats = input.asFloatBuffer()
        for (i in 0 until plane) {
            val p = pixels[i]
            floats.put(i, ((p shr 16) and 0xFF) / 255f)
            floats.put(plane + i, ((p shr 8) and 0xFF) / 255f)
            floats.put(2 * plane + i, (p and 0xFF) / 255f)
        }
        return lb
    }

    override fun close() {
        session.close()
    }

    companion object {
        const val MODEL_ASSET = "model.onnx"
    }
}
