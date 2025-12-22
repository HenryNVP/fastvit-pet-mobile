package com.example.petclassify

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OnnxTensor
import java.awt.image.BufferedImage
import java.io.File
import java.nio.FloatBuffer
import java.util.Collections
import javax.imageio.ImageIO
import kotlin.math.exp

// --- IMPORTANT ---
// Path to the dataset on your LOCAL COMPUTER.
private val DATASET_ROOT_PATH = "C:/fastvit/fastvit-pet-mobile/data/test_app"
// Path to the model file in your project.
private val MODEL_PATH = "../app/src/main/assets/fastvit.onnx"


fun main() {
    val TAG = "FastViT_Local_Evaluation"
    println("🚀 Starting Full Dataset Evaluation on Local JVM...")

    try {
        val env = OrtEnvironment.getEnvironment()
        val session = env.createSession(MODEL_PATH)

        val testAnnotations = File(DATASET_ROOT_PATH, "annotations/test.txt").readLines()
        if (testAnnotations.isEmpty()) {
            println("❌ Error: test.txt is empty or not found at $DATASET_ROOT_PATH/annotations/test.txt")
            return
        }

        val inferenceTimes = mutableListOf<Double>()
        var top1Correct = 0
        var top5Correct = 0

        testAnnotations.forEachIndexed { index, line ->
            val parts = line.split(" ")
            val imageName = parts[0] + ".jpg"
            val trueLabel = parts[1].toInt() - 1 // Class IDs are 1-based in the file

            val imageFile = File(DATASET_ROOT_PATH, "images/$imageName")
            if (!imageFile.exists()) {
                println("⚠️ Image not found, skipping: $imageName")
                return@forEachIndexed
            }

            val image: BufferedImage = ImageIO.read(imageFile)
            // Resize image
            val resizedImage = BufferedImage(256, 256, BufferedImage.TYPE_INT_RGB)
            val g = resizedImage.createGraphics()
            g.drawImage(image, 0, 0, 256, 256, null)
            g.dispose()


            val floatBuffer = convertBitmapToFloatBuffer(resizedImage)
            val inputTensor = OnnxTensor.createTensor(env, floatBuffer, longArrayOf(1, 3, 256, 256))
            val inputs = Collections.singletonMap(session.inputNames.first(), inputTensor)

            val startTime = System.nanoTime()
            val results = session.run(inputs)
            val endTime = System.nanoTime()

            val durationMs = (endTime - startTime) / 1_000_000.0
            inferenceTimes.add(durationMs)

            val outputTensor = results.get(0) as OnnxTensor
            val logits = outputTensor.floatBuffer.array()
            val top5Predictions = getTopKClasses(logits, 5)

            if (top5Predictions.isNotEmpty() && top5Predictions[0].index == trueLabel) {
                top1Correct++
            }
            if (top5Predictions.any { it.index == trueLabel }) {
                top5Correct++
            }

            if ((index + 1) % 100 == 0) {
                println("Processed ${index + 1} / ${testAnnotations.size} images...")
            }
        }

        val totalImages = testAnnotations.size
        val avgInferenceTime = inferenceTimes.average()
        val top1Accuracy = (top1Correct.toDouble() / totalImages) * 100
        val top5Accuracy = (top5Correct.toDouble() / totalImages) * 100

        println("========================================")
        println("✅ Evaluation Complete!")
        println("========================================")
        println("📊 Total Images:      $totalImages")
        println("⏱️ Avg Inference Time: ${"%.2f".format(avgInferenceTime)} ms")
        println("🎯 Top-1 Accuracy:     ${"%.2f".format(top1Accuracy)}%")
        println("🎯 Top-5 Accuracy:     ${"%.2f".format(top5Accuracy)}%")
        println("========================================")

        session.close()
        env.close()

    } catch (e: Exception) {
        println("❌ Error during evaluation: ${e.message}")
        e.printStackTrace()
    }
}

private fun convertBitmapToFloatBuffer(bitmap: BufferedImage): FloatBuffer {
    val imgData = FloatBuffer.allocate(1 * 3 * 256 * 256)
    imgData.rewind()
    val stride = 256 * 256
    val intValues = IntArray(stride)
    bitmap.getRGB(0, 0, 256, 256, intValues, 0, 256)

    val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    val std = floatArrayOf(0.229f, 0.224f, 0.225f)

    for (i in 0 until stride) {
        val pixel = intValues[i]
        // Extract RGB
        val r_val = (pixel shr 16 and 0xFF)
        val g_val = (pixel shr 8 and 0xFF)
        val b_val = (pixel and 0xFF)

        // Normalize and Add to Buffer in NCHW format
        val r = ((r_val / 255.0f - mean[0]) / std[0])
        val g = ((g_val / 255.0f - mean[1]) / std[1])
        val b = ((b_val / 255.0f - mean[2]) / std[2])

        imgData.put(i, r)
        imgData.put(i + stride, g)
        imgData.put(i + stride * 2, b)
    }
    return imgData
}

data class Prediction(val index: Int, val probability: Float)

private fun getTopKClasses(logits: FloatArray, k: Int): List<Prediction> {
    val predictions = logits.mapIndexed { index, logit -> Prediction(index, logit) }
        .sortedByDescending { it.probability }
    return predictions.take(k)
}
