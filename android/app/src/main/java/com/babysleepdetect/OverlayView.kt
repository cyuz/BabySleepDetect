package com.babysleepdetect

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import com.babysleepdetect.detector.Detection
import kotlin.math.max

/** Draws the baby's box and pose skeleton over a PreviewView using FILL_CENTER scaling. */
class OverlayView @JvmOverloads constructor(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {
    private var baby: Detection? = null
    private var imgW = 1
    private var imgH = 1
    private var mirror = false
    private var kptConf = 0.5f

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        isAntiAlias = true
    }
    private val bonePaint = Paint().apply {
        color = Color.CYAN
        strokeWidth = 5f
        isAntiAlias = true
    }
    private val pointPaint = Paint().apply {
        color = Color.YELLOW
        isAntiAlias = true
    }

    fun update(baby: Detection?, imgW: Int, imgH: Int, mirror: Boolean, color: Int, kptConf: Float) {
        this.baby = baby
        this.imgW = imgW
        this.imgH = imgH
        this.mirror = mirror
        this.kptConf = kptConf
        boxPaint.color = color
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val d = baby ?: return
        val scale = max(width.toFloat() / imgW, height.toFloat() / imgH)
        val offX = (width - imgW * scale) / 2f
        val offY = (height - imgH * scale) / 2f
        fun vx(x: Float): Float {
            val v = offX + x * scale
            return if (mirror) width - v else v
        }
        fun vy(y: Float) = offY + y * scale

        canvas.drawRect(
            minOf(vx(d.x1), vx(d.x2)), vy(d.y1), maxOf(vx(d.x1), vx(d.x2)), vy(d.y2), boxPaint,
        )
        val k = d.keypoints ?: return
        for ((a, b) in SKELETON) {
            if (k[a * 3 + 2] >= kptConf && k[b * 3 + 2] >= kptConf) {
                canvas.drawLine(vx(k[a * 3]), vy(k[a * 3 + 1]), vx(k[b * 3]), vy(k[b * 3 + 1]), bonePaint)
            }
        }
        for (i in 0 until k.size / 3) {
            if (k[i * 3 + 2] >= kptConf) canvas.drawCircle(vx(k[i * 3]), vy(k[i * 3 + 1]), 8f, pointPaint)
        }
    }

    companion object {
        private val SKELETON = listOf(
            0 to 1, 0 to 2, 1 to 3, 2 to 4, // face
            5 to 6, 5 to 7, 7 to 9, 6 to 8, 8 to 10, // arms
            5 to 11, 6 to 12, 11 to 12, // torso
            11 to 13, 13 to 15, 12 to 14, 14 to 16, // legs
        )
    }
}
