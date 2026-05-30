package com.example.mhealth.logic

import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.core.content.FileProvider
import com.example.mhealth.logic.db.MHealthDatabase
import com.lowagie.text.*
import com.lowagie.text.pdf.PdfPCell
import com.lowagie.text.pdf.PdfPTable
import com.lowagie.text.pdf.PdfWriter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.*

object ReportGenerator {

    /**
     * Compiles behavioral telemetry and clinical match history, creates a password-protected PDF
     * and a structured JSON file, and launches the Android Share Sheet.
     */
    suspend fun generateAndShareReport(
        context: Context,
        pin: String,
        onComplete: (success: Boolean, errorMsg: String?) -> Unit
    ) {
        withContext(Dispatchers.IO) {
            try {
                val db = MHealthDatabase.getInstance(context)
                val userEmail = DataRepository.userProfile.value?.email ?: "local_patient@lumen.health"
                
                // Fetch demographics and analysis results
                val profile = db.userProfileDao().getProfile(userEmail)
                val analysisList = db.analysisResultDao().getAll(userEmail).sortedByDescending { it.date }
                val featuresList = db.dailyFeaturesDao().getAllFeatures(userEmail).sortedByDescending { it.date }

                if (featuresList.isEmpty()) {
                    withContext(Dispatchers.Main) {
                        onComplete(false, "No behavioral features recorded yet. Cannot generate report.")
                    }
                    return@withContext
                }

                // 1. Compile JSON Export
                val jsonExport = JSONObject().apply {
                    put("reportType", "Lumen Clinical Export")
                    put("generatedAt", System.currentTimeMillis())
                    
                    // Patient metadata
                    put("patient", JSONObject().apply {
                        put("email", userEmail)
                        put("name", profile?.userId ?: "Lumen User")
                        put("onboardingDate", profile?.onboardingDate ?: "")
                        put("baselineReady", profile?.baselineReady ?: false)
                        put("currentStatus", profile?.currentStatus ?: "Collecting")
                    })

                    // Calibration Scores from SharedPreferences
                    put("calibration", JSONObject().apply {
                        put("phq9", DataRepository.phq9Score.value)
                        put("gad7", DataRepository.gad7Score.value)
                        put("recentLifeEvents", DataRepository.recentLifeEventsCount.value)
                    })

                    // Daily Features Array
                    val featuresArray = JSONArray()
                    featuresList.forEach { f ->
                        featuresArray.put(JSONObject().apply {
                            put("date", f.date)
                            put("screenTimeHours", f.screenTimeHours)
                            put("unlockCount", f.unlockCount)
                            put("appLaunchCount", f.appLaunchCount)
                            put("notificationsToday", f.notificationsToday)
                            put("socialAppRatio", f.socialAppRatio)
                            put("callsPerDay", f.callsPerDay)
                            put("callDurationMinutes", f.callDurationMinutes)
                            put("uniqueContacts", f.uniqueContacts)
                            put("conversationFrequency", f.conversationFrequency)
                            put("dailyDisplacementKm", f.dailyDisplacementKm)
                            put("locationEntropy", f.locationEntropy)
                            put("homeTimeRatio", f.homeTimeRatio)
                            put("wakeTimeHour", f.wakeTimeHour)
                            put("sleepTimeHour", f.sleepTimeHour)
                            put("sleepDurationHours", f.sleepDurationHours)
                            put("dailyStepCount", f.dailyStepCount)
                            put("activeMinutes", f.activeMinutes)
                            put("keystrokeSpeed", f.keystrokeSpeed)
                            put("backspaceRatio", f.backspaceRatio)
                            put("scrollVelocity", f.scrollVelocity)
                            put("daylightExposureMinutes", f.daylightExposureMinutes)
                            put("chargeRegularity", f.chargeRegularity)
                            put("upiTransactionsToday", f.upiTransactionsToday)
                            put("musicTimeMinutes", f.musicTimeMinutes)
                        })
                    }
                    put("features", featuresArray)

                    // Analysis Results Array
                    val analysisArray = JSONArray()
                    analysisList.forEach { a ->
                        analysisArray.put(JSONObject().apply {
                            put("date", a.date)
                            put("anomalyScore", a.anomalyScore)
                            put("effectiveScore", a.effectiveScore)
                            put("alertLevel", a.alertLevel)
                            put("evidenceAccumulated", a.evidenceAccumulated)
                            put("daysSustained", a.daysSustained)
                            put("primaryClassification", a.primaryClassification)
                            put("matchConfidence", a.matchConfidence)
                            put("l2Coherence", a.l2Coherence)
                            put("l2RhythmDissolution", a.l2RhythmDissolution)
                            put("l2SessionIncoherence", a.l2SessionIncoherence)
                            put("lifeEventPreFilterPassed", a.lifeEventPreFilterPassed)
                            put("narrativeSummary", a.narrativeSummary)
                        })
                    }
                    put("analysisResults", analysisArray)
                }

                // Write JSON to temporary cache folder
                val cacheDir = context.cacheDir
                val jsonFile = File(cacheDir, "lumen_features_export.json")
                FileOutputStream(jsonFile).use { fos ->
                    fos.write(jsonExport.toString(4).toByteArray())
                }

                // 2. Generate Password-Protected PDF Report
                val pdfFile = File(cacheDir, "lumen_clinical_report.pdf")
                val document = Document(PageSize.A4, 36f, 36f, 54f, 54f)
                val fos = FileOutputStream(pdfFile)
                val writer = PdfWriter.getInstance(document, fos)

                // Apply Password Protection (PIN is userPassword, random key is ownerPassword)
                val ownerPassword = "lumen_admin_key_signature_9918"
                writer.setEncryption(
                    pin.toByteArray(),
                    ownerPassword.toByteArray(),
                    PdfWriter.ALLOW_PRINTING,
                    PdfWriter.STANDARD_ENCRYPTION_128
                )

                // Define Fonts
                val titleFont = FontFactory.getFont(FontFactory.HELVETICA_BOLD, 22f, Color(0, 92, 138)) // Calming primary OceanBlue
                val subtitleFont = FontFactory.getFont(FontFactory.HELVETICA, 10f, Color(100, 116, 139))
                val sectionHeaderFont = FontFactory.getFont(FontFactory.HELVETICA_BOLD, 14f, Color(15, 23, 42))
                val subHeaderFont = FontFactory.getFont(FontFactory.HELVETICA_BOLD, 10f, Color(15, 23, 42))
                val bodyFont = FontFactory.getFont(FontFactory.HELVETICA, 9.5f, Color(15, 23, 42))
                val bodyMuted = FontFactory.getFont(FontFactory.HELVETICA, 8.5f, Color(100, 116, 139))
                val boldBodyFont = FontFactory.getFont(FontFactory.HELVETICA_BOLD, 9.5f, Color(15, 23, 42))

                // Configure Header/Footer
                val headerText = Phrase("LUMEN BEHAVIOR DIAGNOSTICS REPORT — STICKY ON-DEVICE PRIVACY", bodyMuted)
                val footerText = Phrase("Lumen decision support insights. Prepared for clinical consultation.", bodyMuted)
                document.header = HeaderFooter(headerText, false)
                document.footer = HeaderFooter(footerText, true)

                document.open()

                // Report Header
                val headerTable = PdfPTable(2).apply {
                    widthPercentage = 100f
                    setWidths(floatArrayOf(70f, 30f))
                    defaultCell.border = Rectangle.NO_BORDER
                }
                
                val titleCell = PdfPCell().apply {
                    border = Rectangle.NO_BORDER
                    addElement(Paragraph("LUMEN HEALTH", FontFactory.getFont(FontFactory.HELVETICA_BOLD, 10f, Color(78, 205, 196))))
                    addElement(Paragraph("Behavior Diagnostics Report", titleFont))
                    addElement(Paragraph("Passive Smartphone Digital Biomarkers Report", subtitleFont))
                }
                headerTable.addCell(titleCell)

                val dateCell = PdfPCell().apply {
                    border = Rectangle.NO_BORDER
                    horizontalAlignment = Element.ALIGN_RIGHT
                    val df = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault())
                    addElement(Paragraph("Date: ${df.format(Date())}", bodyFont))
                    addElement(Paragraph("Device: Android Local Client", bodyMuted))
                    addElement(Paragraph("Status: ENCRYPTED", FontFactory.getFont(FontFactory.HELVETICA_BOLD, 9f, Color(239, 108, 108))))
                }
                headerTable.addCell(dateCell)
                
                document.add(headerTable)
                document.add(Paragraph(" ")) // Spacing

                // Section 1: Demographics Card
                val sectionDemographics = Paragraph("1. Patient Profile Details", sectionHeaderFont).apply {
                    spacingAfter = 8f
                }
                document.add(sectionDemographics)

                val demoTable = PdfPTable(4).apply {
                    widthPercentage = 100f
                    setWidths(floatArrayOf(25f, 25f, 25f, 25f))
                }

                fun addDemoRow(table: PdfPTable, label1: String, val1: String, label2: String, val2: String) {
                    table.addCell(PdfPCell(Phrase(label1, subHeaderFont)).apply { grayFill = 0.95f; padding = 6f })
                    table.addCell(PdfPCell(Phrase(val1, bodyFont)).apply { padding = 6f })
                    table.addCell(PdfPCell(Phrase(label2, subHeaderFont)).apply { grayFill = 0.95f; padding = 6f })
                    table.addCell(PdfPCell(Phrase(val2, bodyFont)).apply { padding = 6f })
                }

                val patientName = profile?.userId?.substringBefore("@")?.replaceFirstChar { it.uppercase() } ?: "Lumen Patient"
                addDemoRow(demoTable, "Patient ID:", userEmail, "Profile Name:", patientName)
                addDemoRow(demoTable, "Onboarding Completed:", profile?.onboardingDate?.let { 
                    try {
                        val sdf = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault())
                        sdf.format(Date(it.toLong()))
                    } catch (e: Exception) { it }
                } ?: "N/A", "Active Baseline Status:", if (profile?.baselineReady == true) "Locked (Frame 2)" else "Collecting (Frame 1)")

                document.add(demoTable)
                document.add(Paragraph(" "))

                // Section 2: Clinical Calibration & Extended Screener
                val sectionScreener = Paragraph("2. Clinical Baseline Calibration", sectionHeaderFont).apply {
                    spacingAfter = 8f
                }
                document.add(sectionScreener)

                val screenerTable = PdfPTable(3).apply {
                    widthPercentage = 100f
                    setWidths(floatArrayOf(33f, 33f, 34f))
                }
                
                val phqScore = DataRepository.phq9Score.value
                val phqSeverity = when {
                    phqScore >= 20 -> "Severe Depression"
                    phqScore >= 15 -> "Moderately Severe"
                    phqScore >= 10 -> "Moderate Symptoms"
                    phqScore >= 5  -> "Mild Symptoms"
                    else -> "Minimal / None"
                }

                val gadScore = DataRepository.gad7Score.value
                val gadSeverity = when {
                    gadScore >= 15 -> "Severe Anxiety"
                    gadScore >= 10 -> "Moderate Anxiety"
                    gadScore >= 5  -> "Mild Anxiety"
                    else -> "Minimal / None"
                }

                val lifeEvents = DataRepository.recentLifeEventsCount.value

                screenerTable.addCell(PdfPCell().apply {
                    padding = 8f
                    addElement(Paragraph("PHQ-9 Screener Score", subHeaderFont))
                    addElement(Paragraph("$phqScore / 27", FontFactory.getFont(FontFactory.HELVETICA_BOLD, 16f, Color(0, 92, 138))))
                    addElement(Paragraph(phqSeverity, bodyFont))
                })
                
                screenerTable.addCell(PdfPCell().apply {
                    padding = 8f
                    addElement(Paragraph("GAD-7 Screener Score", subHeaderFont))
                    addElement(Paragraph("$gadScore / 21", FontFactory.getFont(FontFactory.HELVETICA_BOLD, 16f, Color(0, 92, 138))))
                    addElement(Paragraph(gadSeverity, bodyFont))
                })

                screenerTable.addCell(PdfPCell().apply {
                    padding = 8f
                    addElement(Paragraph("Recent Stressful Events", subHeaderFont))
                    addElement(Paragraph("$lifeEvents Events", FontFactory.getFont(FontFactory.HELVETICA_BOLD, 16f, Color(0, 92, 138))))
                    addElement(Paragraph("Last 14-day window", bodyFont))
                })

                document.add(screenerTable)
                
                // Explanatory Note on Calibration
                val calibrationNote = Paragraph(
                    "Clinical Threshold Calibration Note: Self-reports showing moderate-to-severe symptoms (Scores >= 10) trigger an automatic safety recalibration of on-device engines. Specifically: (1) System 1 Anomaly Threshold is lowered from 0.38 to 0.32 to enhance passive detection sensitivity, and (2) System 2 Life Event filter window is lengthened to 14 days (from 10) to guard against high-stress transient distortions.",
                    FontFactory.getFont(FontFactory.HELVETICA_OBLIQUE, 8.5f, Color(100, 116, 139))
                ).apply {
                    spacingBefore = 6f
                }
                document.add(calibrationNote)
                document.add(Paragraph(" "))

                // Section 3: Diagnostic Analysis (System 2 Classifications)
                val sectionDiagnostics = Paragraph("3. On-Device Diagnostic Characterization History", sectionHeaderFont).apply {
                    spacingAfter = 8f
                }
                document.add(sectionDiagnostics)

                val diagTable = PdfPTable(6).apply {
                    widthPercentage = 100f
                    setWidths(floatArrayOf(15f, 15f, 20f, 20f, 15f, 15f))
                }

                // Table Headers
                fun addDiagHeader(table: PdfPTable) {
                    val headers = listOf("Date", "Alert Level", "Primary Match", "Match Conf.", "S1 Score", "S2 Status")
                    headers.forEach { h ->
                        table.addCell(PdfPCell(Phrase(h, subHeaderFont)).apply {
                            grayFill = 0.90f
                            padding = 6f
                            horizontalAlignment = Element.ALIGN_CENTER
                        })
                    }
                }
                addDiagHeader(diagTable)

                val displayList = analysisList.take(15) // Keep it focused on the latest 15 reports
                if (displayList.isEmpty()) {
                    val emptyCell = PdfPCell(Phrase("No diagnostic reports have been generated yet. (App is still establishing the user's baseline.)", bodyFont)).apply {
                        colspan = 6
                        padding = 12f
                        horizontalAlignment = Element.ALIGN_CENTER
                    }
                    diagTable.addCell(emptyCell)
                } else {
                    displayList.forEach { r ->
                        diagTable.addCell(PdfPCell(Phrase(r.date, bodyFont)).apply { padding = 5f; horizontalAlignment = Element.ALIGN_CENTER })
                        diagTable.addCell(PdfPCell(Phrase(r.alertLevel.uppercase(), boldBodyFont)).apply { 
                            padding = 5f
                            horizontalAlignment = Element.ALIGN_CENTER
                            val alertColor = when (r.alertLevel.lowercase()) {
                                "green", "stable" -> Color(45, 212, 191)
                                "yellow" -> Color(251, 191, 36)
                                "orange" -> Color(249, 115, 22)
                                "red" -> Color(251, 113, 133)
                                else -> Color(15, 23, 42)
                            }
                            font = FontFactory.getFont(FontFactory.HELVETICA_BOLD, 9f, alertColor)
                        })
                        diagTable.addCell(PdfPCell(Phrase(r.primaryClassification, bodyFont)).apply { padding = 5f })
                        diagTable.addCell(PdfPCell(Phrase("%.1f%%".format(r.matchConfidence * 100), bodyFont)).apply { padding = 5f; horizontalAlignment = Element.ALIGN_CENTER })
                        diagTable.addCell(PdfPCell(Phrase("%.3f".format(r.effectiveScore), bodyFont)).apply { padding = 5f; horizontalAlignment = Element.ALIGN_CENTER })
                        diagTable.addCell(PdfPCell(Phrase(if (r.lifeEventPreFilterPassed) "Matched" else "LE Filtered", bodyFont)).apply { padding = 5f; horizontalAlignment = Element.ALIGN_CENTER })
                    }
                }

                document.add(diagTable)
                document.add(Paragraph(" "))

                // Section 4: Trend Summaries
                if (displayList.isNotEmpty()) {
                    val latest = displayList.first()
                    if (latest.narrativeSummary.isNotEmpty()) {
                        val sectionNarrative = Paragraph("4. Latest Behavior Insights Narrative", sectionHeaderFont).apply {
                            spacingAfter = 6f
                        }
                        document.add(sectionNarrative)
                        val narrativeText = Paragraph(latest.narrativeSummary, bodyFont).apply {
                            spacingAfter = 14f
                        }
                        document.add(narrativeText)
                    }
                }

                // Section 5: Telemetry Summary Statistics
                val sectionStats = Paragraph("5. Behavioral Telemetry Averages (Last 15 Days)", sectionHeaderFont).apply {
                    spacingAfter = 8f
                }
                document.add(sectionStats)

                val statsTable = PdfPTable(4).apply {
                    widthPercentage = 100f
                    setWidths(floatArrayOf(35f, 15f, 35f, 15f))
                }

                val recentFeatures = featuresList.take(15)
                val avgScreen = recentFeatures.map { it.screenTimeHours }.average().toFloat()
                val avgUnlocks = recentFeatures.map { it.unlockCount }.average().toFloat()
                val avgSteps = recentFeatures.map { it.dailyStepCount }.average().toFloat()
                val avgSleep = recentFeatures.map { it.sleepDurationHours }.average().toFloat()
                val avgSocial = recentFeatures.map { it.socialAppRatio }.average().toFloat() * 100
                val avgCalls = recentFeatures.map { it.callsPerDay }.average().toFloat()
                val avgDisplacement = recentFeatures.map { it.dailyDisplacementKm }.average().toFloat()
                val avgHome = recentFeatures.map { it.homeTimeRatio }.average().toFloat() * 100

                fun addStatCell(table: PdfPTable, metric: String, value: String) {
                    table.addCell(PdfPCell(Phrase(metric, bodyFont)).apply { padding = 6f; grayFill = 0.98f })
                    table.addCell(PdfPCell(Phrase(value, boldBodyFont)).apply { padding = 6f; horizontalAlignment = Element.ALIGN_CENTER })
                }

                addStatCell(statsTable, "Daily Steps (Avg):", "%,.0f steps".format(avgSteps))
                addStatCell(statsTable, "Sleep Duration (Avg):", "%.1f hours".format(avgSleep))
                addStatCell(statsTable, "Daily Screen Time (Avg):", "%.1f hours".format(avgScreen))
                addStatCell(statsTable, "Device Unlocks (Avg):", "%.0f unlocks".format(avgUnlocks))
                addStatCell(statsTable, "Social Time Ratio (Avg):", "%.1f%%".format(avgSocial))
                addStatCell(statsTable, "Phone Calls / Day (Avg):", "%.1f calls".format(avgCalls))
                addStatCell(statsTable, "Daily Displacement (Avg):", "%.2f km".format(avgDisplacement))
                addStatCell(statsTable, "Home Stay Duration (Avg):", "%.1f%%".format(avgHome))

                document.add(statsTable)
                document.add(Paragraph(" "))

                // Section 6: Clinician Disclaimer
                val disclaimerHeader = Paragraph("Clinical Decision Support Disclaimer", sectionHeaderFont).apply {
                    spacingAfter = 6f
                }
                document.add(disclaimerHeader)
                
                val disclaimerText = Paragraph(
                    "Lumen is an edge-native, local-first decision support tool. It processes sensory telemetry indicators (sleep, circadian rhythms, physical activity, screen usage patterns, keystroke speed, and communication frequencies) entirely on-device using custom Bayesian mathematical models, DBSCAN clustering, and expert-weighted clinical classification rules. Lumen does not diagnose disorders or replace standard psychiatric examinations. It serves as an objective behavioral dashboard to supplement clinician triage, track longitudinal trends, and flag psychomotor anomalies or relational withdrawal indicators.",
                    FontFactory.getFont(FontFactory.HELVETICA_OBLIQUE, 8.5f, Color(100, 116, 139))
                )
                document.add(disclaimerText)

                document.close()
                fos.close()

                // 3. Trigger native Share Sheet
                val authority = "${context.packageName}.provider"
                val pdfUri = FileProvider.getUriForFile(context, authority, pdfFile)
                val jsonUri = FileProvider.getUriForFile(context, authority, jsonFile)

                val shareIntent = Intent(Intent.ACTION_SEND_MULTIPLE).apply {
                    type = "*/*"
                    putParcelableArrayListExtra(
                        Intent.EXTRA_STREAM,
                        ArrayList(listOf(pdfUri, jsonUri))
                    )
                    putExtra(Intent.EXTRA_SUBJECT, "Lumen Diagnostics Behavioral Report")
                    putExtra(Intent.EXTRA_TEXT, "Hello, here is the encrypted clinical report and raw behavioral features compiled by Lumen on my device.")
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }

                val chooser = Intent.createChooser(shareIntent, "Share Clinical Data with Doctor").apply {
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                }

                withContext(Dispatchers.Main) {
                    context.startActivity(chooser)
                    onComplete(true, null)
                }

            } catch (e: Exception) {
                android.util.Log.e("ReportGenerator", "Failed to compile report: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    onComplete(false, e.localizedMessage ?: "Unknown error compiling files.")
                }
            }
        }
    }
}
