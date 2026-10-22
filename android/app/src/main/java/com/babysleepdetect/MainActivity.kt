package com.babysleepdetect

import android.Manifest
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.os.Bundle
import android.os.SystemClock
import android.text.InputType
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.babysleepdetect.detector.FrameResult
import com.babysleepdetect.detector.PostureModel
import com.babysleepdetect.posture.MonitorConfig
import com.babysleepdetect.posture.MonitorState
import com.babysleepdetect.posture.Posture
import com.babysleepdetect.posture.PostureConfig
import com.babysleepdetect.posture.SleepMonitor
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var statusText: TextView
    private lateinit var detailText: TextView
    private lateinit var muteButton: Button

    private lateinit var prefs: SharedPreferences
    private lateinit var analysisExecutor: ExecutorService
    private lateinit var alarm: Alarm
    private val notifier = WebhookNotifier()
    private val monitor = SleepMonitor()

    @Volatile private var model: PostureModel? = null
    @Volatile private var postureCfg = PostureConfig()
    private var useFrontCamera = false
    private var muted = false

    private val requestCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else showStatus(getString(R.string.need_camera), Color.DKGRAY)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.preview)
        overlay = findViewById(R.id.overlay)
        statusText = findViewById(R.id.status)
        detailText = findViewById(R.id.detail)
        muteButton = findViewById(R.id.mute)

        prefs = getSharedPreferences("settings", MODE_PRIVATE)
        alarm = Alarm(this)
        applySettings()

        findViewById<Button>(R.id.switch_camera).setOnClickListener {
            useFrontCamera = !useFrontCamera
            prefs.edit().putBoolean(KEY_FRONT, useFrontCamera).apply()
            startCamera()
        }
        findViewById<Button>(R.id.settings).setOnClickListener { showSettings() }
        muteButton.setOnClickListener {
            muted = true
            alarm.stop()
        }

        analysisExecutor = Executors.newSingleThreadExecutor()
        analysisExecutor.execute {
            try {
                model = PostureModel(this)
            } catch (e: Exception) {
                Log.e(TAG, "model load failed", e)
                runOnUiThread { showStatus(getString(R.string.model_missing), Color.DKGRAY) }
            }
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            requestCamera.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        alarm.stop()
        analysisExecutor.execute { model?.close() }
        analysisExecutor.shutdown()
    }

    private fun applySettings() {
        useFrontCamera = prefs.getBoolean(KEY_FRONT, false)
        postureCfg = PostureConfig(kptConf = prefs.getFloat(KEY_KPT_CONF, 0.5f))
        monitor.cfg = MonitorConfig(
            alertAfterSeconds = prefs.getFloat(KEY_ALERT_AFTER, 5f).toDouble(),
            missingAfterSeconds = prefs.getFloat(KEY_MISSING_AFTER, 30f).toDouble(),
        )
        notifier.url = prefs.getString(KEY_WEBHOOK, "") ?: ""
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val resolution = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .build()
            val preview = Preview.Builder().setResolutionSelector(resolution).build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(resolution)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(analysisExecutor, ::analyze) }
            val selector = if (useFrontCamera) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
            try {
                provider.unbindAll()
                provider.bindToLifecycle(this, selector, preview, analysis)
            } catch (e: Exception) {
                Log.e(TAG, "camera bind failed", e)
                Toast.makeText(this, R.string.camera_failed, Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(image: ImageProxy) {
        val m = model
        if (m == null) {
            image.close()
            return
        }
        val frame = image.use { rotate(it.toBitmap(), it.imageInfo.rotationDegrees) }
        val result = m.run(frame, CONF, postureCfg)
        val state = monitor.update(result.result.posture, SystemClock.elapsedRealtime() / 1000.0)
        val front = useFrontCamera
        runOnUiThread { render(result, state, frame.width, frame.height, front) }
    }

    private fun rotate(bitmap: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return bitmap
        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun render(frame: FrameResult, state: MonitorState, w: Int, h: Int, front: Boolean) {
        val color = when {
            state.alert -> COLOR_ALERT
            else -> POSTURE_COLORS.getValue(state.posture)
        }
        overlay.update(frame.baby, w, h, front, color, postureCfg.kptConf)

        val text = when {
            state.alert -> getString(R.string.alert_prone, state.proneSeconds.toInt())
            state.missing -> getString(R.string.missing)
            state.posture == Posture.PRONE -> getString(R.string.status_prone, state.proneSeconds.toInt())
            else -> getString(R.string.status_posture, state.posture.zh)
        }
        showStatus(text, color)
        detailText.text = getString(
            R.string.detail, frame.result.posture.zh, frame.result.reason, frame.inferenceMs,
        )

        if (state.alert) {
            if (!muted) alarm.start()
            notifier.notify(getString(R.string.alert_prone, state.proneSeconds.toInt()))
        } else {
            alarm.stop()
            muted = false
            if (state.missing) notifier.notify(getString(R.string.missing))
        }
        muteButton.isEnabled = alarm.playing
    }

    private fun showStatus(text: String, color: Int) {
        statusText.text = text
        statusText.setBackgroundColor(color)
    }

    private fun showSettings() {
        fun field(hint: Int, value: String, type: Int) = EditText(this).apply {
            setHint(hint)
            setText(value)
            inputType = type
        }
        val decimal = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
        val alertAfter = field(R.string.setting_alert_after, prefs.getFloat(KEY_ALERT_AFTER, 5f).toString(), decimal)
        val missingAfter = field(R.string.setting_missing_after, prefs.getFloat(KEY_MISSING_AFTER, 30f).toString(), decimal)
        val kptConf = field(R.string.setting_kpt_conf, prefs.getFloat(KEY_KPT_CONF, 0.5f).toString(), decimal)
        val webhook = field(R.string.setting_webhook, prefs.getString(KEY_WEBHOOK, "") ?: "", InputType.TYPE_TEXT_VARIATION_URI)

        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 16, 48, 0)
            for ((label, view) in listOf(
                R.string.setting_alert_after to alertAfter,
                R.string.setting_missing_after to missingAfter,
                R.string.setting_kpt_conf to kptConf,
                R.string.setting_webhook to webhook,
            )) {
                addView(TextView(context).apply { setText(label) })
                addView(view)
            }
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.settings)
            .setView(layout)
            .setPositiveButton(android.R.string.ok) { _, _ ->
                prefs.edit()
                    .putFloat(KEY_ALERT_AFTER, alertAfter.text.toString().toFloatOrNull() ?: 5f)
                    .putFloat(KEY_MISSING_AFTER, missingAfter.text.toString().toFloatOrNull() ?: 30f)
                    .putFloat(KEY_KPT_CONF, (kptConf.text.toString().toFloatOrNull() ?: 0.5f).coerceIn(0.05f, 0.95f))
                    .putString(KEY_WEBHOOK, webhook.text.toString().trim())
                    .apply()
                applySettings()
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    companion object {
        private const val TAG = "BabySleep"
        private const val CONF = 0.25f
        private const val KEY_FRONT = "front_camera"
        private const val KEY_ALERT_AFTER = "alert_after"
        private const val KEY_MISSING_AFTER = "missing_after"
        private const val KEY_KPT_CONF = "kpt_conf"
        private const val KEY_WEBHOOK = "webhook"

        private val COLOR_ALERT = Color.rgb(198, 40, 40)
        private val POSTURE_COLORS = mapOf(
            Posture.SUPINE to Color.rgb(46, 125, 50),
            Posture.SIDE to Color.rgb(249, 168, 37),
            Posture.PRONE to Color.rgb(230, 81, 0),
            Posture.UNKNOWN to Color.rgb(85, 85, 85),
        )
    }
}
