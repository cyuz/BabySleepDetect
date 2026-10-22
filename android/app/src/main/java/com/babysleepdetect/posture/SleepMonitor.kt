package com.babysleepdetect.posture

/**
 * Temporal smoothing and alert logic on top of per-frame postures.
 * Kotlin port of babysleep/monitor.py; keep the two in sync.
 */
data class MonitorConfig(
    val windowSeconds: Double = 3.0, // smoothing window for the per-frame votes
    val alertAfterSeconds: Double = 5.0, // prone must persist this long before alarming
    val missingAfterSeconds: Double = 30.0, // warn if no baby seen this long (0 disables)
    val proneRatio: Double = 0.6, // fraction of prone frames in the window to count as prone
)

data class MonitorState(
    val posture: Posture,
    val alert: Boolean,
    val proneSeconds: Double,
    val missing: Boolean,
)

class SleepMonitor(var cfg: MonitorConfig = MonitorConfig()) {
    private val votes = ArrayDeque<Pair<Double, Posture>>()
    private var proneSince: Double? = null
    private var lastSeen: Double? = null

    fun update(posture: Posture, t: Double): MonitorState {
        if (lastSeen == null || posture != Posture.UNKNOWN) lastSeen = t

        votes.addLast(t to posture)
        while (votes.isNotEmpty() && t - votes.first().first > cfg.windowSeconds) votes.removeFirst()

        val smoothed = smoothed()
        if (smoothed == Posture.PRONE) {
            if (proneSince == null) proneSince = t
        } else {
            proneSince = null
        }

        val proneSeconds = proneSince?.let { t - it } ?: 0.0
        val missing = cfg.missingAfterSeconds > 0 && t - lastSeen!! >= cfg.missingAfterSeconds
        return MonitorState(smoothed, proneSeconds >= cfg.alertAfterSeconds, proneSeconds, missing)
    }

    private fun smoothed(): Posture {
        val known = votes.map { it.second }.filter { it != Posture.UNKNOWN }
        if (known.isEmpty()) return Posture.UNKNOWN
        if (known.count { it == Posture.PRONE }.toDouble() / known.size >= cfg.proneRatio) return Posture.PRONE
        return known.groupingBy { it }.eachCount().maxBy { it.value }.key
    }
}
