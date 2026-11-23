package com.kittentts.pico
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
object AudioUtils {
    fun saveWavToStream(audio: FloatArray, out: OutputStream, sampleRate: Int) {
        val dataSize = audio.size * 2
        val b = ByteBuffer.allocate(44 + dataSize).order(ByteOrder.LITTLE_ENDIAN)
        b.put("RIFF".toByteArray()); b.putInt(dataSize + 36); b.put("WAVE".toByteArray())
        b.put("fmt ".toByteArray()); b.putInt(16); b.putShort(1); b.putShort(1)
        b.putInt(sampleRate); b.putInt(sampleRate * 2); b.putShort(2); b.putShort(16)
        b.put("data".toByteArray()); b.putInt(dataSize)
        for (s in audio) b.putShort((s.coerceIn(-1.0f, 1.0f) * 32767).toInt().toShort())
        out.write(b.array()); out.flush()
    }
    fun saveWav(audio: FloatArray, file: File, sampleRate: Int) {
        FileOutputStream(file).use { saveWavToStream(audio, it, sampleRate) }
    }
}