package com.example.petclassify

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OnnxTensor
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.util.Collections
import android.graphics.Bitmap
import android.graphics.BitmapFactory

class MainActivity : AppCompatActivity() {

    // --- IMPORTANT --- 
    // YOU MUST CHANGE THIS PATH to the absolute path of the dataset on your device.
    private val DATASET_ROOT_PATH = "/storage/emulated/0/Download/test_app"

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { _ ->
        // This block is called when the user returns from the settings screen.
        // We just re-check the permission.
        checkAndRequestPermission()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // No UI needed. We just need to handle permissions and then start the work.
    }

    override fun onResume() {
        super.onResume()
        // Check permission every time the app comes to the foreground.
        checkAndRequestPermission()
    }

    private fun checkAndRequestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) { // Android 11+
            if (Environment.isExternalStorageManager()) {
                // All files access is already granted.
                startEvaluation()
            } else {
                // All files access is not granted.
                Toast.makeText(this, "Please grant 'All files access' permission for this app.", Toast.LENGTH_LONG).show()
                // Create an intent to open the settings screen for our app.
                val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION)
                val uri = Uri.fromParts("package", packageName, null)
                intent.data = uri
                requestPermissionLauncher.launch(intent)
            }
        } else {
            // For older Android versions, the manifest permission should be enough.
            // If it still fails, the user needs to grant it from settings manually.
            startEvaluation()
        }
    }

    private fun startEvaluation() {
        // Run in a background thread to avoid blocking the UI.
        Thread {
            runInferenceAndLogResults()
        }.start()
    }

    private fun copyAssetsToCache() {
        val assetManager = assets
        val files = assetManager.list("")
        files?.forEach { filename ->
            if (filename.endsWith(".onnx") || filename.endsWith(".data")) {
                val destFile = File(cacheDir, filename)
                if (!destFile.exists()) {
                    try {
                        assetManager.open(filename).use { inStream ->
                            FileOutputStream(destFile).use { outStream ->
                                inStream.copyTo(outStream)
                            }
                        }
                        Log.d("OfflineEval", "Copied $filename to cache.")
                    } catch (e: Exception) {
                        Log.e("OfflineEval", "Failed to copy $filename", e)
                    }
                }
            }
        }
    }

    private fun runInferenceAndLogResults() {
        val TAG = "OfflineEval"
        val RESULT_TAG = "EVAL_RESULT"

        // Check if we've already run this to prevent re-running on every onResume.
        if (isEvaluationRunning) return
        isEvaluationRunning = true

        Log.d(TAG, "🚀 Starting Inference... Will log results for offline analysis.")

        try {
            val env = OrtEnvironment.getEnvironment()
            copyAssetsToCache()
            val modelPath = File(cacheDir, "fastvit.onnx").absolutePath
            val session = env.createSession(modelPath)

            // Corrected the path to the annotations file.
            val testAnnotations = File(DATASET_ROOT_PATH, "test.txt").readLines()
            if (testAnnotations.isEmpty()) {
                Log.e(TAG, "❌ Error: test.txt is empty or not found at $DATASET_ROOT_PATH/test.txt")
                return
            }

            testAnnotations.forEachIndexed { index, line ->
                val parts = line.split(" ")
                val imageName = parts[0]
                val trueLabel = parts[1].toInt() - 1 // Class IDs are 1-based

                val imageFile = File(DATASET_ROOT_PATH, "images/${imageName}.jpg")
                if (!imageFile.exists()) {
                    Log.w(TAG, "⚠️ Image not found, skipping: ${imageName}.jpg")
                    return@forEachIndexed
                }

                val bitmap = BitmapFactory.decodeFile(imageFile.absolutePath)
                val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 256, 256, true)

                val floatBuffer = convertBitmapToFloatBuffer(resizedBitmap)
                val inputTensor = OnnxTensor.createTensor(env, floatBuffer, longArrayOf(1, 3, 256, 256))
                val inputs = Collections.singletonMap(session.inputNames.first(), inputTensor)

                val startTime = System.nanoTime()
                val results = session.run(inputs)
                val endTime = System.nanoTime()
                val durationMs = (endTime - startTime) / 1_000_000.0

                val outputTensor = results.get(0) as OnnxTensor
                val logits = outputTensor.floatBuffer.array()

                val top5Indices = getTopKIndices(logits, 5)

                val logLine = "$trueLabel,${durationMs},${top5Indices.joinToString(",")}"
                Log.d(RESULT_TAG, logLine)

                if ((index + 1) % 100 == 0) {
                    Log.d(TAG, "Processed ${index + 1} / ${testAnnotations.size} images...")
                }
            }

            Log.i(TAG, "✅ Finished processing all images.")

            session.close()
            env.close()

        } catch (e: Exception) {
            Log.e(TAG, "❌ Error during inference: ${e.message}")
            e.printStackTrace()
        }
    }

    private fun convertBitmapToFloatBuffer(bitmap: Bitmap): FloatBuffer {
        val imgData = FloatBuffer.allocate(1 * 3 * 256 * 256)
        imgData.rewind()
        val stride = 256 * 256
        val intValues = IntArray(stride)
        bitmap.getPixels(intValues, 0, 256, 0, 0, 256, 256)
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)

        for (i in 0 until stride) {
            val pixel = intValues[i]
            val r = (((pixel shr 16) and 0xFF) / 255.0f - mean[0]) / std[0]
            val g = (((pixel shr 8) and 0xFF) / 255.0f - mean[1]) / std[1]
            val b = ((pixel and 0xFF) / 255.0f - mean[2]) / std[2]
            imgData.put(i, r)
            imgData.put(i + stride, g)
            imgData.put(i + stride * 2, b)
        }
        return imgData
    }

    private fun getTopKIndices(logits: FloatArray, k: Int): List<Int> {
        return logits.mapIndexed { index, logit -> index to logit }
            .sortedByDescending { it.second }
            .take(k)
            .map { it.first }
    }

    companion object {
        // A simple flag to prevent the evaluation from running multiple times.
        private var isEvaluationRunning = false
    }
}
