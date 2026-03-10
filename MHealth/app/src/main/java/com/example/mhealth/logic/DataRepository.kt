package com.example.mhealth.logic

import com.example.mhealth.models.DailyReport
import com.example.mhealth.models.PersonalityVector
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

object DataRepository {
    private val _latestVector = MutableStateFlow<PersonalityVector?>(null)
    val latestVector: StateFlow<PersonalityVector?> = _latestVector

    private val _reports = MutableStateFlow<List<DailyReport>>(emptyList())
    val reports: StateFlow<List<DailyReport>> = _reports

    private val _baseline = MutableStateFlow<PersonalityVector?>(null)
    val baseline: StateFlow<PersonalityVector?> = _baseline

    private val _isBuildingBaseline = MutableStateFlow(true)
    val isBuildingBaseline: StateFlow<Boolean> = _isBuildingBaseline

    private val _baselineProgress = MutableStateFlow(0) // Days collected
    val baselineProgress: StateFlow<Int> = _baselineProgress

    fun updateLatestVector(vector: PersonalityVector) {
        _latestVector.value = vector
    }

    fun addReport(report: DailyReport) {
        _reports.value = _reports.value + report
    }

    fun setBaseline(vector: PersonalityVector) {
        _baseline.value = vector
        _isBuildingBaseline.value = false
    }

    fun updateBaselineProgress(days: Int) {
        _baselineProgress.value = days
    }
}
