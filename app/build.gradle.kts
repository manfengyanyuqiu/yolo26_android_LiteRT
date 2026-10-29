plugins { id("com.android.application") }
android {
    namespace = "com.fydnn.dimensity"
    compileSdk = 36
    buildToolsVersion = "36.0.0"
    defaultConfig {
        applicationId = "com.fydnn.dimensity"
        minSdk = 31
        targetSdk = 36
        versionCode = 11
        versionName = "0.6.4"
        testInstrumentationRunner = "com.fydnn.dimensity.NpuDeviceInstrumentation"
        ndk { abiFilters += "arm64-v8a" }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    packaging { jniLibs.useLegacyPackaging = true }
}
dependencies {
    implementation("com.google.ai.edge.litert:litert:2.3.0")
    implementation("com.google.ai.edge.litert:litert-npu-runtime-mediatek:2.3.0")
    implementation("org.jetbrains.kotlin:kotlin-stdlib:2.1.20")
    testImplementation("junit:junit:4.13.2")
}
