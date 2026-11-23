
package com.kittentts.pico

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.json.JSONObject
import java.nio.FloatBuffer
import java.nio.IntBuffer

class KittenTtsEngine(private val context: Context) {
    private val env = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private var voiceEmbeddings: Map<String, FloatArray> = emptyMap()
    private var g2pDict: Map<String, List<Int>> = emptyMap()

    fun initialize() {
        try {
            val g2pObj = JSONObject(context.assets.open("cmu_misaki_us.json").bufferedReader().use { it.readText() })
            val g2p = mutableMapOf<String, List<Int>>()
            val keys = g2pObj.keys()
            while (keys.hasNext()) {
                val key = keys.next()
                val arr = g2pObj.optJSONArray(key)
                if (arr != null) g2p[key] = (0 until arr.length()).map { i -> arr.getInt(i) }
            }
            g2pDict = g2p

            val loaded = mutableMapOf<String, FloatArray>()
            val data = context.assets.open("voices_en.npz").readBytes()
            val zis = java.util.zip.ZipInputStream(java.io.ByteArrayInputStream(data))
            var entry = zis.nextEntry
            while (entry != null) {
                if (entry.name.endsWith(".npy")) {
                    loaded[entry.name.removeSuffix(".npy")] = parseFloatArrayFromNpy(zis.readBytes())
                }
                entry = zis.nextEntry
            }
            zis.close()
            voiceEmbeddings = loaded

            val modelBytes = context.assets.open("kitten_tts_2_pico_en_int4.onnx").readBytes()
            session = env.createSession(modelBytes, OrtSession.SessionOptions())
        } catch (e: Exception) {
            throw RuntimeException("Failed to initialize TTS: ${e.message}", e)
        }
    }

    private fun parseFloatArrayFromNpy(data: ByteArray): FloatArray {
        var headerEnd = -1
        for (i in 0 until minOf(data.size, 10000)) {
            if (data[i] == 10.toByte()) { headerEnd = i + 1; break }
        }
        if (headerEnd < 0) return FloatArray(256)
        val count = (data.size - headerEnd) / 4
        val floats = FloatArray(count)
        for (i in 0 until count) {
            val o = headerEnd + i * 4
            if (o + 4 <= data.size) {
                val bits = ((data[o].toInt() and 0xFF) or ((data[o+1].toInt() and 0xFF) shl 8) or
                            ((data[o+2].toInt() and 0xFF) shl 16) or ((data[o+3].toInt() and 0xFF) shl 24))
                floats[i] = Float.fromBits(bits)
            }
        }
        return floats
    }

    fun generate(text: String, progressCallback: (Int) -> Unit = {}): FloatArray {
        progressCallback(10)
        val phonemeIds = textToPhonemes(text)
        progressCallback(30)
        val voice = voiceEmbeddings.values.firstOrNull() ?: FloatArray(256) { 0.0f }
        progressCallback(50)
        val audio = runInference(phonemeIds, voice)
        progressCallback(90)
        val maxVal = audio.maxOfOrNull { Math.abs(it) } ?: 1.0f
        val scale = if (maxVal == 0.0f) 1.0f else 0.95f / maxVal
        progressCallback(100)
        return audio.map { it * scale }.toFloatArray()
    }

    private fun textToPhonemes(text: String): IntArray {
        val phonemes = mutableListOf<Int>()
        for (ch in text.lowercase()) {
            val ids = g2pDict[ch.toString()]
            if (ids != null) phonemes.addAll(ids)
            else if (ch.isLetterOrDigit() || ch == ' ') phonemes.add(ch.code % 100)
        }
        if (phonemes.isEmpty()) phonemes.add(0)
        return phonemes.toIntArray()
    }

    private fun runInference(phonemeIds: IntArray, voice: FloatArray): FloatArray {
        val sess = session ?: throw RuntimeException("Session not initialized")
        return try {
            val pTensor = OnnxTensor.createTensor(env, IntBuffer.wrap(phonemeIds), longArrayOf(1, phonemeIds.size.toLong()))
            val vTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(voice), longArrayOf(1, voice.size.toLong()))
            val results = sess.run(mapOf("phonemes" to pTensor, "voice" to vTensor))
            val output = (results.get(0).value as OnnxTensor).floatBuffer.array().clone()
            pTensor.close(); vTensor.close(); results.close()
            output
        } catch (e: Exception) {
            FloatArray(24000) { 0.0f }
        }
    }

    fun close() { try { session?.close() } catch (_: Exception) {} }
}
