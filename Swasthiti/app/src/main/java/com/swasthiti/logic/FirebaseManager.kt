package com.swasthiti.logic

import android.util.Log
import com.google.firebase.firestore.FirebaseFirestore
import kotlinx.coroutines.tasks.await

/**
 * Systematic Manager for Firebase operations.
 * Centralizes Firestore instance and provides connectivity checks.
 */
object FirebaseManager {
    private const val TAG = "FirebaseManager"
    
    // Provide a systematic way to get the Firestore instance
    val db: FirebaseFirestore by lazy {
        val instance = FirebaseFirestore.getInstance()
        
        // Simple settings: Disable persistence to force real network checks
        val settings = com.google.firebase.firestore.firestoreSettings {
            setLocalCacheSettings(com.google.firebase.firestore.MemoryCacheSettings.newBuilder().build())
        }
        
        instance.firestoreSettings = settings
        instance
    }

    /**
     * Verifies the linkage with Firestore.
     * Hits a "ping" collection to see if the database is reachable.
     */
    suspend fun checkDatabaseLinkage(): Boolean {
        return try {
            Log.d(TAG, "🔍 [TRACE 1] System: Checking connectivity...")
            
            val results = with(kotlinx.coroutines.Dispatchers.IO) {
                // 🌐 Test 1: Google.com
                val googleOk = try {
                    val conn = java.net.URL("https://www.google.com").openConnection() as java.net.HttpURLConnection
                    conn.connectTimeout = 3000
                    conn.connect()
                    conn.responseCode == 200
                } catch (e: Exception) { false }
                
                // 🔥 Test 2: Firestore REST API (Experimental)
                // This checks if we can reach the database using standard web protocols
                val restOk = try {
                    val projectId = "swasthiti-10" // Matches your provided google-services.json
                    val url = "https://firestore.googleapis.com/v1/projects/$projectId/databases/(default)/documents/system/handshake"
                    val conn = java.net.URL(url).openConnection() as java.net.HttpURLConnection
                    conn.connectTimeout = 5000
                    conn.connect()
                    val code = conn.responseCode
                    Log.d(TAG, "🔍 [TRACE 1.2] System: Firestore REST code = $code")
                    code == 200 || code == 404 // 404 means reached but doc missing, which is still a "connected" state
                } catch (e: Exception) {
                    Log.e(TAG, "❌ [TRACE 1.2 ERROR] System: Cannot reach Firestore via REST! ${e.message}")
                    false
                }
                Pair(googleOk, restOk)
            }

            Log.d(TAG, "🔍 [TRACE 2] FirebaseManager: Starting SDK handshake...")
            val dbInstance = db
            
            val versionDoc = dbInstance.collection("system")
                .document("handshake")
                .get(com.google.firebase.firestore.Source.SERVER)
                .await()
            
            val exists = versionDoc.exists()
            Log.d(TAG, "🔍 [TRACE 3] FirebaseManager: SDK Handshake result: exists=$exists")
            exists
        } catch (e: Exception) {
            Log.e(TAG, "❌ [TRACE 4 ERROR] FirebaseManager: SDK Check failed!")
            Log.e(TAG, "❌ Reason: ${e.message}", e)
            
            if (e is com.google.firebase.firestore.FirebaseFirestoreException) {
                Log.e(TAG, "❌ Firestore Error Code: ${e.code}")
            }
            false
        }
    }
}
