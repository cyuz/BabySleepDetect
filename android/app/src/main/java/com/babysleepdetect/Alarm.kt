package com.babysleepdetect

import android.content.Context
import android.media.AudioAttributes
import android.media.Ringtone
import android.media.RingtoneManager
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Log
import java.net.HttpURLConnection
import java.net.URL
import kotlin.concurrent.thread

/** Loud looping alarm sound (alarm stream, ignores media volume) plus vibration. */
class Alarm(context: Context) {
    private val ringtone: Ringtone? = RingtoneManager.getRingtone(
        context,
        RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM)
            ?: RingtoneManager.getDefaultUri(RingtoneManager.TYPE_RINGTONE),
    )?.apply {
        audioAttributes = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_ALARM)
            .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
            .build()
        isLooping = true
    }
    private val vibrator: Vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        (context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager).defaultVibrator
    } else {
        @Suppress("DEPRECATION")
        context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
    }

    var playing = false
        private set

    fun start() {
        if (playing) return
        playing = true
        ringtone?.play()
        vibrator.vibrate(VibrationEffect.createWaveform(longArrayOf(0, 600, 400), 0))
    }

    fun stop() {
        if (!playing) return
        playing = false
        ringtone?.stop()
        vibrator.cancel()
    }
}

/** POSTs a JSON message to a webhook (ntfy, Discord, Home Assistant …), rate-limited. */
class WebhookNotifier(private val cooldownMs: Long = 60_000) {
    @Volatile var url: String = ""
    private var last = 0L

    fun notify(message: String) {
        val target = url.trim()
        val now = System.currentTimeMillis()
        if (target.isEmpty() || now - last < cooldownMs) return
        last = now
        thread(isDaemon = true) {
            try {
                val conn = URL(target).openConnection() as HttpURLConnection
                conn.requestMethod = "POST"
                conn.connectTimeout = 5000
                conn.readTimeout = 5000
                conn.doOutput = true
                conn.setRequestProperty("Content-Type", "application/json; charset=utf-8")
                val escaped = message.replace("\\", "\\\\").replace("\"", "\\\"")
                conn.outputStream.use { it.write("""{"text":"$escaped","content":"$escaped"}""".toByteArray()) }
                conn.responseCode
                conn.disconnect()
            } catch (e: Exception) {
                Log.w("BabySleep", "webhook failed", e)
            }
        }
    }
}
