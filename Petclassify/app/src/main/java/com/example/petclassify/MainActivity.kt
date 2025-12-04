package com.example.petclassify

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OnnxTensor
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.util.Collections
import kotlin.math.exp

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // No UI needed. Check Logcat for results.
        Thread {
            runInference()
        }.start()
    }

    /**
     * Copies files from the assets folder to the app's cache directory.
     * This is necessary for ONNX models that have external data files.
     */
    private fun copyAssetsToCache() {
        val assetManager = assets
        val files = assetManager.list("")
        if (files != null) {
            for (filename in files) {
                // Copy only ONNX model and data files.
                if (filename.endsWith(".onnx") || filename.endsWith(".data")) {
                    val destFile = File(cacheDir, filename)
                    try {
                        assetManager.open(filename).use { inStream ->
                            FileOutputStream(destFile).use { outStream ->
                                inStream.copyTo(outStream)
                            }
                        }
                        Log.d("FastViT_Inference", "Copied $filename to cache.")
                    } catch (e: Exception) {
                        Log.e("FastViT_Inference", "Failed to copy $filename", e)
                    }
                }
            }
        }
    }

    private fun runInference() {
        val TAG = "FastViT_Inference"
        Log.d(TAG, "🚀 Starting Inference...")

        try {
            val env = OrtEnvironment.getEnvironment()

            // 1. Load Model
            // Copy assets to cache to get a file path, required for models with external data.
            copyAssetsToCache()
            val modelPath = File(cacheDir, "fastvit.onnx").absolutePath
            val session = env.createSession(modelPath)

            val imagePath = "shiba_inu_images"
            val imageFileNames = assets.list(imagePath)?.filter { it.endsWith(".jpg") } ?: emptyList()
            if (imageFileNames.isEmpty()) {
                Log.e(TAG, "No JPG images found in assets/shiba_inu_images folder.")
                return
            }

            val inferenceTimes = mutableListOf<Double>()

            for (imageFileName in imageFileNames) {
                val fullImagePath = "$imagePath/$imageFileName"
                // 2. Load & Preprocess Image
                // FastViT typically uses 256x256 input
                val bitmap = BitmapFactory.decodeStream(assets.open(fullImagePath))
                val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 256, 256, true)

                val floatBuffer = convertBitmapToFloatBuffer(resizedBitmap)
                val inputTensor = OnnxTensor.createTensor(env, floatBuffer, longArrayOf(1, 3, 256, 256))
                val inputs = Collections.singletonMap(session.inputNames.first(), inputTensor)

                // 3. Run Inference
                Log.d(TAG, "📸 Processing image: $fullImagePath")
                val startTime = System.nanoTime()

                val results = session.run(inputs)

                val endTime = System.nanoTime()
                val durationMs = (endTime - startTime) / 1_000_000.0
                inferenceTimes.add(durationMs)

                // 4. Parse Output (Top 1)
                val outputTensor = results.get(0) as OnnxTensor
                val floatArray = outputTensor.floatBuffer.array() // Raw logits

                val topClass = getTopClass(floatArray)

                Log.i(TAG, "========================================")
                Log.i(TAG, "🖼️ Image: $fullImagePath")
                Log.i(TAG, "🐶 RESULT: Class Index ${topClass.index}")
                Log.i(TAG, "🧠 Confidence: ${"%.2f".format(topClass.probability * 100)}%")
                Log.i(TAG, "⏱️ Time: ${"%.2f".format(durationMs)} ms")
                Log.i(TAG, "========================================")

                // Note: Index 242 is usually 'Boxer' in ImageNet
                if (topClass.index == 242) Log.i(TAG, "✅ Correctly identified as Boxer!")
            }

            val averageInferenceTime = inferenceTimes.average()
            Log.i(TAG, "========================================")
            Log.i(TAG, "📊 Average Inference Time: ${"%.2f".format(averageInferenceTime)} ms over ${inferenceTimes.size} images.")
            Log.i(TAG, "========================================")


            session.close()
            env.close()

        } catch (e: Exception) {
            Log.e(TAG, "❌ Error: ${e.message}")
            e.printStackTrace()
        }
    }

    // --- Helper: Preprocessing (Standard ImageNet Normalization) ---
    private fun convertBitmapToFloatBuffer(bitmap: Bitmap): FloatBuffer {
        val imgData = FloatBuffer.allocate(1 * 3 * 256 * 256)
        imgData.rewind()

        val stride = 256 * 256
        val intValues = IntArray(stride)
        bitmap.getPixels(intValues, 0, 256, 0, 0, 256, 256)

        // Standard ImageNet Mean & Std
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)

        // Loop through pixels and convert to Planar format (RRR...GGG...BBB...)
        // This is much faster than doing it pixel-by-pixel inside the loop
        for (i in 0 until 256 * 256) {
            val pixel = intValues[i]

            // Extract RGB (ignore Alpha)
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            // Normalize and Add to Buffer (Planar layout: all Rs, then all Gs, then all Bs)
            // But we need to write to specific indices if using a single loop,
            // OR simple approach: write sequentially if we trust the buffer order.
            // ONNX expects NCHW (Batch, Channel, Height, Width).
            // It's safer to fill separate arrays first or calculate offset.

            // Let's use 3 passes or calculated index to ensure NCHW layout
        }

        // Optimized NCHW filler
        for (i in 0 until 256 * 256) {
            val pixel = intValues[i]
            val r = (((pixel shr 16) and 0xFF) / 255.0f - mean[0]) / std[0]
            imgData.put(i, r) // Red channel at start

            val g = (((pixel shr 8) and 0xFF) / 255.0f - mean[1]) / std[1]
            imgData.put(i + stride, g) // Green channel in middle

            val b = ((pixel and 0xFF) / 255.0f - mean[2]) / std[2]
            imgData.put(i + stride * 2, b) // Blue channel at end
        }

        return imgData
    }

    // --- Helper: Softmax & Argmax ---
    data class Prediction(val index: Int, val probability: Float)

    private fun getTopClass(logits: FloatArray): Prediction {
        var maxIndex = -1
        var maxLogit = -Float.MAX_VALUE

        // Find max to prevent overflow during exp
        for (i in logits.indices) {
            if (logits[i] > maxLogit) {
                maxLogit = logits[i]
                maxIndex = i
            }
        }

        // Calculate Softmax for the top class only
        var sumExp = 0.0f
        for (logit in logits) {
            sumExp += exp(logit - maxLogit)
        }
        val probability = exp(logits[maxIndex] - maxLogit) / sumExp

        return Prediction(maxIndex, probability)
    }
}
