Run ./gradlew assembleRelease --no-daemon
To honour the JVM settings for this build a single-use Daemon process will be forked. For more on this, please refer to https://docs.gradle.org/8.11.1/userguide/gradle_daemon.html#sec:disabling_the_daemon in the Gradle documentation.
Daemon JVM discovery is an incubating feature.
Daemon will be stopped at the end of the build 

> Configure project :app
Warning: Python version 3.11 may have fewer packages available. If you experience problems, try switching to version 3.8.

> Task :app:buildKotlinToolingMetadata
> Task :app:checkKotlinGradlePluginConfigurationErrors SKIPPED
> Task :app:preBuild UP-TO-DATE
> Task :app:preReleaseBuild UP-TO-DATE
> Task :app:generateReleaseBuildConfig
> Task :app:extractReleasePythonBuildPackages

> Task :app:generateReleasePythonRequirements
Chaquopy: Installing for arm64-v8a
Looking in indexes: https://pypi.org/simple, https://chaquo.com/pypi-13.1
Collecting numpy==1.26.2
  Using cached https://chaquo.com/pypi-13.1/numpy/numpy-1.26.2-0-cp311-cp311-android_21_arm64_v8a.whl (5.0 MB)
Collecting pandas==2.1.3
  Using cached https://chaquo.com/pypi-13.1/pandas/pandas-2.1.3-1-cp311-cp311-android_24_arm64_v8a.whl (11.3 MB)
Collecting python-dateutil==2.9.0
  Using cached python_dateutil-2.9.0-py2.py3-none-any.whl (230 kB)
Collecting six==1.17.0
  Using cached six-1.17.0-py2.py3-none-any.whl (11 kB)
Collecting pytz==2024.1
  Using cached pytz-2024.1-py2.py3-none-any.whl (505 kB)
Collecting chaquopy-libcxx>=11000
  Using cached https://chaquo.com/pypi-13.1/chaquopy-libcxx/chaquopy_libcxx-180000-0-py3-none-android_24_arm64_v8a.whl (413 kB)
Collecting chaquopy-openblas>=0.2.20
  Using cached https://chaquo.com/pypi-13.1/chaquopy-openblas/chaquopy_openblas-0.2.20-5-py3-none-android_21_arm64_v8a.whl (4.3 MB)
Collecting tzdata>=2022.1
  Using cached tzdata-2026.2-py2.py3-none-any.whl (349 kB)
Collecting chaquopy-libgfortran>=4.9
  Using cached https://chaquo.com/pypi-13.1/chaquopy-libgfortran/chaquopy_libgfortran-4.9-0-py3-none-android_21_arm64_v8a.whl (495 kB)
Installing collected packages: chaquopy-libcxx, chaquopy-libgfortran, chaquopy-openblas, numpy, pytz, six, python-dateutil, tzdata, pandas
Successfully installed chaquopy-libcxx-180000 chaquopy-libgfortran-4.9 chaquopy-openblas-0.2.20 numpy-1.26.2 pandas-2.1.3 python-dateutil-2.9.0 pytz-2024.1 six-1.17.0 tzdata-2026.2
Chaquopy: Installing for armeabi-v7a
Looking in indexes: https://pypi.org/simple, https://chaquo.com/pypi-13.1
Collecting chaquopy-libcxx==180000
  Using cached https://chaquo.com/pypi-13.1/chaquopy-libcxx/chaquopy_libcxx-180000-0-py3-none-android_24_armeabi_v7a.whl (355 kB)
Collecting chaquopy-libgfortran==4.9
  Using cached https://chaquo.com/pypi-13.1/chaquopy-libgfortran/chaquopy_libgfortran-4.9-0-py3-none-android_16_armeabi_v7a.whl (394 kB)
Collecting chaquopy-openblas==0.2.20
  Using cached https://chaquo.com/pypi-13.1/chaquopy-openblas/chaquopy_openblas-0.2.20-5-py3-none-android_16_armeabi_v7a.whl (4.0 MB)
Collecting numpy==1.26.2
  Using cached https://chaquo.com/pypi-13.1/numpy/numpy-1.26.2-0-cp311-cp311-android_21_armeabi_v7a.whl (5.0 MB)
Collecting pandas==2.1.3
  Using cached https://chaquo.com/pypi-13.1/pandas/pandas-2.1.3-1-cp311-cp311-android_24_armeabi_v7a.whl (11.1 MB)
Installing collected packages: chaquopy-libcxx, chaquopy-libgfortran, chaquopy-openblas, numpy, pandas
Successfully installed chaquopy-libcxx-180000 chaquopy-libgfortran-4.9 chaquopy-openblas-0.2.20 numpy-1.26.2 pandas-2.1.3

> Task :app:mergeReleasePythonSources
> Task :app:generateReleasePythonProxies
> Task :app:generateReleaseResValues
> Task :app:checkReleaseAarMetadata
> Task :app:processReleaseGoogleServices
> Task :app:mapReleaseSourceSetPaths
> Task :app:generateReleaseResources
> Task :app:packageReleaseResources
> Task :app:createReleaseCompatibleScreenManifests
> Task :app:extractDeepLinksRelease

> Task :app:processReleaseMainManifest
package="com.example.mhealth" found in source AndroidManifest.xml: /home/runner/work/Mental-Health-Detection-ML/Mental-Health-Detection-ML/MHealth - Copy/app/src/main/AndroidManifest.xml.
Setting the namespace via the package attribute in the source AndroidManifest.xml is no longer supported, and the value is ignored.
Recommendation: remove package="com.example.mhealth" from the source AndroidManifest.xml: /home/runner/work/Mental-Health-Detection-ML/Mental-Health-Detection-ML/MHealth - Copy/app/src/main/AndroidManifest.xml.

> Task :app:processReleaseManifest
> Task :app:processReleaseManifestForPackage
> Task :app:parseReleaseLocalResources
> Task :app:extractProguardFiles
> Task :app:javaPreCompileRelease
> Task :app:mergeReleaseResources
> Task :app:generateReleasePythonJniLibs
> Task :app:mergeReleaseJniLibFolders
> Task :app:mergeReleaseNativeLibs
> Task :app:checkReleaseDuplicateClasses

> Task :app:stripReleaseDebugSymbols
Unable to strip the following libraries, packaging them as they are: libandroidx.graphics.path.so, libchaquopy_java.so, libcrypto_chaquopy.so, libcrypto_python.so, libpython3.11.so, libsqlite3_chaquopy.so, libsqlite3_python.so, libssl_chaquopy.so, libssl_python.so.

> Task :app:mergeReleaseStartupProfile
> Task :app:mergeReleaseArtProfile
> Task :app:extractReleaseNativeSymbolTables
> Task :app:mergeReleaseShaders
> Task :app:mergeReleaseNativeDebugMetadata NO-SOURCE
> Task :app:compileReleaseShaders NO-SOURCE
> Task :app:generateReleaseAssets UP-TO-DATE
> Task :app:processReleaseResources
> Task :app:generateReleasePythonMiscAssets
> Task :app:generateReleasePythonRequirementsAssets
> Task :app:generateReleasePythonSourceAssets
> Task :app:generateReleasePythonBuildAssets
> Task :app:mergeReleaseAssets
> Task :app:extractReleaseVersionControlInfo
> Task :app:compressReleaseAssets
> Task :app:collectReleaseDependencies
> Task :app:sdkReleaseDependencyData
> Task :app:validateSigningRelease
> Task :app:writeReleaseAppMetadata
> Task :app:writeReleaseSigningConfigVersions
> Task :app:convertLinkedResourcesToProtoRelease
> Task :app:kspReleaseKotlin

e: file:///home/runner/work/Mental-Health-Detection-ML/Mental-Health-Detection-ML/MHealth%20-%20Copy/app/src/release/java/com/example/mhealth/MainActivity.kt:760:29 @Composable invocations can only happen from the context of a @Composable function
> Task :app:compileReleaseKotlin FAILED
e: file:///home/runner/work/Mental-Health-Detection-ML/Mental-Health-Detection-ML/MHealth%20-%20Copy/app/src/release/java/com/example/mhealth/MainActivity.kt:761:32 @Composable invocations can only happen from the context of a @Composable function
e: file:///home/runner/work/Mental-Health-Detection-ML/Mental-Health-Detection-ML/MHealth%20-%20Copy/app/src/release/java/com/example/mhealth/MainActivity.kt:762:29 @Composable invocations can only happen from the context of a @Composable function

[Incubating] Problems report is available at: file:///home/runner/work/Mental-Health-Detection-ML/Mental-Health-Detection-ML/MHealth%20-%20Copy/build/reports/problems/problems-report.html
FAILURE: Build failed with an exception.


Deprecated Gradle features were used in this build, making it incompatible with Gradle 9.0.
* What went wrong:

Execution failed for task ':app:compileReleaseKotlin'.
You can use '--warning-mode all' to show the individual deprecation warnings and determine if they come from your own scripts or plugins.
> A failure occurred while executing org.jetbrains.kotlin.compilerRunner.GradleCompilerRunnerWithWorkers$GradleKotlinCompilerWorkAction

   > Compilation error. See log for more details
For more on this, please refer to https://docs.gradle.org/8.11.1/userguide/command_line_interface.html#sec:command_line_warnings in the Gradle documentation.

46 actionable tasks: 46 executed
* Try:
> Run with --stacktrace option to get the stack trace.
> Run with --info or --debug option to get more log output.
> Run with --scan to get full insights.
> Get more help at https://help.gradle.org.

BUILD FAILED in 1m 4s
Error: Process completed with exit code 1.