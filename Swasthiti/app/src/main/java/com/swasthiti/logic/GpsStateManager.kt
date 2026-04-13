package com.swasthiti.logic

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.util.Log
import com.swasthiti.models.LatLonPoint
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlin.math.sqrt

/**
 * Adaptive GPS State Machine
 *
 * Automatically adjusts GPS polling interval based on user activity:
 * - STATIONARY: 30-min interval (desk, home, no movement)
 * - WALKING: 5-min interval (pedestrian movement)
 * - VEHICLE: 2-min interval (car, bus, train - speed > 6 km/h)
 *
 * State transitions use step counter + GPS speed fusion for fast, accurate detection.
 */
enum class GpsState(val intervalMs: Long, val accuracyThreshold: Float, val displayName: String) {
    STATIONARY(30 * 60_000L, 200f, "Stationary"),  // 30 min, ≤200m filter
    WALKING(5 * 60_000L, 100f, "Walking"),         // 5 min, ≤100m filter
    VEHICLE(2 * 60_000L, 50f, "Vehicle")           // 2 min, ≤50m filter
}

class GpsStateManager(private val context: Context) : SensorEventListener {

    private val TAG = "Swasthiti.GpsStateManager"

    private val sensorManager = context.getSystemService(SensorManager::class.java)
    private var stepSensor: Sensor? = null

    // Current GPS state
    private val _currentState = MutableStateFlow(GpsState.STATIONARY)
    val currentState: StateFlow<GpsState> = _currentState

    // Step counter tracking
    private var lastStepCount = 0f
    private var stepCheckStartCount = 0f
    private var lastStepCheckTime = 0L

    // GPS speed tracking for vehicle detection
    private var lastGpsSpeed = 0f
    private var vehicleSpeedStartTime = 0L
    private var isVehicleSpeedSustained = false

    // State transition debouncing
    private var stationarySince = 0L
    private var walkingSince = 0L

    // Configuration
    private val WALK_SPEED_MS = 6.0 / 3.6  // 5.5 km/h - anything above is vehicle
    private val VEHICLE_SPEED_MS = 8.0 / 3.6  // 8 km/h - clear vehicle indicator
    private val VEHICLE_SUSTAIN_MS = 3 * 60_000L  // 3 min sustained = confirmed vehicle
    private val STEPS_FOR_MOVEMENT = 10  // steps in 2 min = walking
    private val STEP_CHECK_WINDOW_MS = 2 * 60_000L  // 2 min window

    init {
        registerStepSensor()
    }

    /** Register for step counter sensor (hardware-based, low power). */
    private fun registerStepSensor() {
        stepSensor = sensorManager?.getDefaultSensor(Sensor.TYPE_STEP_COUNTER)
        if (stepSensor != null) {
            sensorManager?.registerListener(this, stepSensor, SensorManager.SENSOR_DELAY_NORMAL)
            Log.i(TAG, "Step counter registered for adaptive GPS")
        } else {
            Log.w(TAG, "Step counter not available - using GPS-only detection")
        }
    }

    /** Unregister step sensor on cleanup. */
    fun unregister() {
        sensorManager?.unregisterListener(this)
    }

    /** Update state based on new GPS fix. */
    fun onGpsFixReceived(fix: LatLonPoint) {
        val speedMs = fix.speed

        // Check for vehicle speed
        if (speedMs > VEHICLE_SPEED_MS) {
            if (vehicleSpeedStartTime == 0L) {
                vehicleSpeedStartTime = System.currentTimeMillis()
            } else if (System.currentTimeMillis() - vehicleSpeedStartTime >= VEHICLE_SUSTAIN_MS) {
                isVehicleSpeedSustained = true
            }
        } else {
            vehicleSpeedStartTime = 0L
            isVehicleSpeedSustained = false
        }

        // Transition logic
        when (_currentState.value) {
            GpsState.STATIONARY -> {
                if (isVehicleSpeedSustained) {
                    transitionTo(GpsState.VEHICLE, "sustained vehicle speed ${speedMs * 3.6f} km/h")
                }
            }
            GpsState.WALKING -> {
                if (isVehicleSpeedSustained) {
                    transitionTo(GpsState.VEHICLE, "sustained vehicle speed ${speedMs * 3.6f} km/h")
                }
            }
            GpsState.VEHICLE -> {
                if (speedMs < WALK_SPEED_MS && !isVehicleSpeedSustained) {
                    // Will transition back to walking if no steps detected
                    checkTransitionFromVehicle()
                }
            }
        }

        lastGpsSpeed = speedMs
    }

    /** Check if we should transition from VEHICLE back to WALKING/STATIONARY. */
    private fun checkTransitionFromVehicle() {
        // If speed dropped and no new steps, wait a bit before transitioning
        vehicleSpeedStartTime = 0L
        isVehicleSpeedSustained = false
        // Transition will happen on next step check if no movement
    }

    /** Transition to a new state with logging. */
    private fun transitionTo(newState: GpsState, reason: String) {
        if (_currentState.value != newState) {
            Log.i(TAG, "GPS State: ${_currentState.value.displayName} → ${newState.displayName} ($reason)")
            _currentState.value = newState
            when (newState) {
                GpsState.STATIONARY -> stationarySince = System.currentTimeMillis()
                GpsState.WALKING -> walkingSince = System.currentTimeMillis()
                GpsState.VEHICLE -> {}
            }
        }
    }

    // =========================================================================
    // SensorEventListener - Step counter drives state transitions
    // =========================================================================

    override fun onSensorChanged(event: SensorEvent?) {
        if (event?.sensor?.type != Sensor.TYPE_STEP_COUNTER) return

        val currentSteps = event.values[0]
        val now = System.currentTimeMillis()

        // Handle step counter wrap-around (resets at 1M steps)
        val stepDelta = if (currentSteps < lastStepCount) {
            currentSteps  // Counter wrapped
        } else {
            currentSteps - lastStepCount
        }

        lastStepCount = currentSteps

        // Initialize step check window
        if (lastStepCheckTime == 0L) {
            stepCheckStartCount = currentSteps
            lastStepCheckTime = now
            return
        }

        // Check steps in 2-min window
        val windowSteps = currentSteps - stepCheckStartCount
        val windowElapsed = now - lastStepCheckTime

        if (windowElapsed >= STEP_CHECK_WINDOW_MS) {
            // Evaluate state based on step count
            when (_currentState.value) {
                GpsState.STATIONARY -> {
                    if (windowSteps >= STEPS_FOR_MOVEMENT) {
                        transitionTo(GpsState.WALKING, "$windowSteps steps in 2 min")
                    }
                }
                GpsState.WALKING -> {
                    if (windowSteps < STEPS_FOR_MOVEMENT) {
                        // Check if we've been stationary for a while
                        if (stationarySince == 0L) stationarySince = now
                        else if (now - stationarySince >= 10 * 60_000L) {
                            transitionTo(GpsState.STATIONARY, "no movement for 10 min")
                        }
                    } else {
                        stationarySince = 0L  // Reset stationary timer
                    }
                }
                GpsState.VEHICLE -> {
                    if (windowSteps >= STEPS_FOR_MOVEMENT) {
                        // User is walking, not in vehicle
                        transitionTo(GpsState.WALKING, "steps detected while in vehicle state")
                    } else if (windowSteps < STEPS_FOR_MOVEMENT) {
                        // No steps + low speed = stationary
                        if (stationarySince == 0L) stationarySince = now
                        else if (now - stationarySince >= 10 * 60_000L) {
                            transitionTo(GpsState.STATIONARY, "no steps for 10 min after vehicle")
                        }
                    }
                }
            }

            // Reset window
            stepCheckStartCount = currentSteps
            lastStepCheckTime = now
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Not used for step counter
    }

    /** Get current polling interval for location request. */
    fun getCurrentIntervalMs(): Long = _currentState.value.intervalMs

    /** Get accuracy threshold for current state. */
    fun getCurrentAccuracyThreshold(): Float = _currentState.value.accuracyThreshold

    /** Force a state transition (for testing/debugging). */
    fun forceState(state: GpsState) {
        Log.i(TAG, "Force state: ${state.displayName}")
        _currentState.value = state
    }

    /** Reset state to STATIONARY (call on day reset). */
    fun reset() {
        lastStepCount = 0f
        stepCheckStartCount = 0f
        lastStepCheckTime = 0L
        vehicleSpeedStartTime = 0L
        isVehicleSpeedSustained = false
        stationarySince = 0L
        walkingSince = 0L
        forceState(GpsState.STATIONARY)
    }
}

