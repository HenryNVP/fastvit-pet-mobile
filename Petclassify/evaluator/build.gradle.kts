plugins {
    id("org.jetbrains.kotlin.jvm")
}

dependencies {
    implementation(kotlin("stdlib-jdk8"))
    implementation("com.microsoft.onnxruntime:onnxruntime:1.17.1")
}

// Task to run the evaluation script
tasks.register<JavaExec>("runEvaluation") {
    mainClass.set("com.example.petclassify.LocalEvaluationKt")
    classpath = sourceSets.getByName("main").runtimeClasspath
}
