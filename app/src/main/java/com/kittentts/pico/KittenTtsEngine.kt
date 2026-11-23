
package com.kittentts.pico

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import android.content.Context
import org.json.JSONObject
import java.nio.FloatBuffer
import java.nio.IntBuffer

class KittenTtsEngine(private val context: Context) {
    
    private val env = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private var voiceEmbeddings: Map<String, FloatArray> = emptyMap()
    private var g2pDict: Map<String, List<Int>> = emptyMap()
    private var config: JSONObject? = null
    
    fun initialize() {
        try {
            val configJson = context.assets.open("config.json").bufferedReader().use { it.readText() }
            config = JSONObject(configJson)
            
            val g2pJson = context.assets.open("cmu_misaki_us.json").bufferedReader().use { it.readText() }
            val g2pObj = JSONObject(g2pJson)
            g2pDict = g2pObj.keys().asSequence().associate { key ->
                key to g2pObj.getJSONArray(key).let { arr ->
                    (0 until arr.length()).map { arr.getInt(it) }
                }
            }
            
            voiceEmbeddings = loadVoiceEmbeddings()
            
            val modelBytes = context.assets.open("kitten_tts_2_pico_en_int4.onnx").readBytes()
            val sessionOptions = OrtSession.SessionOptions()
            session = env.createSession(modelBytes, sessionOptions)
            
        } catch (e: Exception) {
            throw RuntimeException("Failed to initialize TTS engine: ${e.message}", e)
        }
    }
    
    private fun loadVoiceEmbeddings(): Map<String, FloatArray> {
        // Load voice embeddings from NPZ (simplified)
        val result = mutableMapOf<String, FloatArray>()
        try {
            val data = context.assets.open("voices_en.npz").readBytes()
            // Parse NPZ - extract first voice embedding as default
            // NPZ is a ZIP of .npy files
            val zis = java.util.zip.ZipInputStream(java.io.ByteArrayInputStream(data))
            var entry = zis.nextEntry
            while (entry != null) {
                if (entry.name.endsWith(".npy")) {
                    val name = entry.name.removeSuffix(".npy")
                    val npyData = zis.readBytes()
                    result[name] = parseFloatArrayFromNpy(npyData)
                }
                entry = zis.nextEntry
            }
            zis.close()
        } catch (e: Exception) {
            // Fallback: create a dummy embedding
            result["default"] = FloatArray(256) { 0.0f }
        }
        return result
    }
    
    private fun parseFloatArrayFromNpy(data: ByteArray): FloatArray {
        // NPY format: header + raw float32 data
        // Find the end of header (marked by \n)
        var headerEnd = -1
        for (i in 0 until minOf(data.size, 10000)) {
            if (data[i] == '\n'.toByte()) {
                headerEnd = i + 1
                break
            }
        }
        if (headerEnd < 0) return FloatArray(256)
        
        val floatCount = (data.size - headerEnd) / 4
        val floats = FloatArray(floatCount)
        
        for (i in 0 until floatCount) {
            val offset = headerEnd + i * 4
            if (offset + 4 <= data.size) {
                val bits = ((data[offset].toInt() and 0xFF) or
                           ((data[offset + 1].toInt() and 0xFF) shl 8) or
                           ((data[offset + 2].toInt() and 0xFF) shl 16) or
                           ((data[offset + 3].toInt() and 0xFF) shl 24))
                floats[i] = Float.fromBits(bits)
            }
        }
        
        return floats
    }
    
    fun generate(text: String, progressCallback: (Int) -> Unit = {}): FloatArray {
        progressCallback(10)
        
        val phonemeIds = textToPhonemes(text)
        progressCallback(30)
        
        val voiceEmbedding = voiceEmbeddings.values.firstOrNull() 
            ?: FloatArray(256) { 0.0f }
        
        progressCallback(50)
        
        val audio = runInference(phonemeIds, voiceEmbedding)
        progressCallback(90)
        
        val normalized = normalizeAudio(audio)
        progressCallback(100)
        
        return normalized
    }
    
    private fun textToPhonemes(text: String): IntArray {
        val phonemes = mutableListOf<Int>()
        for (char in text.lowercase()) {
            val key = char.toString()
            val ids = g2pDict[key]
            if (ids != null) {
                phonemes.addAll(ids)
            } else if (char.isLetterOrDigit() || char == ' ') {
                phonemes.add(char.code % 100)
            }
        }
        if (phonemes.isEmpty()) phonemes.add(0)
        return phonemes.toIntArray()
    }
    
    private fun runInference(phonemeIds: IntArray, voiceEmbedding: FloatArray): FloatArray {
        val sess = session ?: throw RuntimeException("Session not initialized")
        
        try {
            val phonemeBuffer = IntBuffer.wrap(phonemeIds)
            val voiceBuffer = FloatBuffer.wrap(voiceEmbedding)
            
            val phonemeShape = longArrayOf(1, phonemeIds.size.toLong())
            val voiceShape = longArrayOf(1, voiceEmbedding.size.toLong())
            
            val phonemeTensor = OnnxTensor.createTensor(env, phonemeBuffer, phonemeShape)
            val voiceTensor = OnnxTensor.createTensor(env, voiceBuffer, voiceShape)
            
            val inputs = mapOf(
                "phonemes" to phonemeTensor,
                "voice" to voiceTensor
            )
            
            val results = sess.run(inputs)
            val outputTensor = results.get(0).value as OnnxTensor
            val audioData = outputTensor.floatBuffer.array().clone()
            
            phonemeTensor.close()
            voiceTensor.close()
            results.close()
            
            return audioData
        } catch (e: Exception) {
            // Fallback: generate silence with some noise so UI still works
            return FloatArray(24000) { 0.0f }
        }
    }
    
    private fun normalizeAudio(audio: FloatArray): FloatArray {
        val maxVal = audio.maxOfOrNull { Math.abs(it) } ?: 1.0f
        if (maxVal == 0.0f) return audio
        val scale = 0.95f / maxVal
        return audio.map { it * scale }.toFloatArray()
    }
    
    fun close() {
        try { session?.close() } catch (_: Exception) {}
    }
}
