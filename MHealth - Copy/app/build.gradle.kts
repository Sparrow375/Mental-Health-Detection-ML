plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.ksp)
    alias(libs.plugins.chaquopy)
    alias(libs.plugins.google.services)
}

android {
    namespace = "com.example.mhealth"
    compileSdk = 35

    signingConfigs {
        create("sharedDebug") {
            storeFile = file("debug.keystore")
            storePassword = "android"
            keyAlias = "androiddebugkey"
            keyPassword = "android"
        }
        create("releaseConfig") {
            val keystoreFile = System.getenv("RELEASE_STORE_FILE")?.let { file(it) }
                ?: if (file("release.keystore").exists()) file("release.keystore") else file("debug.keystore")
            storeFile = keystoreFile
            storePassword = System.getenv("RELEASE_STORE_PASSWORD") ?: "lumen123"
            keyAlias = System.getenv("RELEASE_KEY_ALIAS") ?: "lumen_key"
            keyPassword = System.getenv("RELEASE_KEY_PASSWORD") ?: "lumen123"
        }
    }

    defaultConfig {
        val isRelease = project.gradle.startParameter.taskNames.any { it.contains("release", ignoreCase = true) }
        applicationId = if (isRelease) "com.lumen.mh.app" else "com.example.mhealth"
        minSdk = 26
        targetSdk = 35

        val baseVersionCode = 3
        val runNumber = System.getenv("GITHUB_RUN_NUMBER")?.toIntOrNull() ?: 0
        versionCode = baseVersionCode + runNumber
        versionName = "1.1.$versionCode"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }
    }

    buildTypes {
        debug {
            signingConfig = signingConfigs.getByName("sharedDebug")
            buildConfigField("boolean", "IS_DEV_BUILD", "true")
        }
        release {
            buildConfigField("boolean", "IS_DEV_BUILD", "false")
            isMinifyEnabled = true
            isShrinkResources = true
            signingConfig = signingConfigs.getByName("releaseConfig")
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    buildFeatures {
        compose = true
        buildConfig = true
    }
}

chaquopy {
    defaultConfig {
        version = "3.11"
        pip {
            install("numpy==1.26.2")
            install("pandas==2.1.3")
            install("python-dateutil==2.9.0")
            install("six==1.17.0")
            install("pytz==2024.1")
        }
    }
}

ksp {
    arg("room.schemaLocation", "$projectDir/schemas")
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.material3.adaptive.navigation.suite)
    implementation(libs.androidx.compose.material.icons.extended)
    implementation(libs.play.services.location)

    // Room (local SQLite persistence)
    implementation(libs.room.runtime)
    implementation(libs.room.ktx)
    ksp(libs.room.compiler)

    // WorkManager (nightly analysis scheduling)
    implementation(libs.workmanager.ktx)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    debugImplementation(libs.androidx.compose.ui.tooling)
    debugImplementation(libs.androidx.compose.ui.test.manifest)

    debugImplementation(platform(libs.firebase.bom))
    debugImplementation(libs.firebase.auth)
    debugImplementation(libs.firebase.firestore)

    // Lottie for animations
    implementation("com.airbnb.android:lottie-compose:6.4.0")

    // OpenPDF for medical reports
    implementation("com.github.librepdf:openpdf:2.0.3")
}
