
package com.kittentts.pico

import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

object AudioUtils {
    fun saveWavToStream(audio: FloatArray, out: OutputStream, sampleRate: Int) {
        val dataSize = audio.size * 2
        val buffer = ByteBuffer.allocate(44 + dataSize).order(ByteOrder.LITTLE_ENDIAN)
        buffer.put("RIFF".toByteArray()); buffer.putInt(dataSize + 36); buffer.put("WAVE".toByteArray())
        buffer.put("fmt ".toByteArray()); buffer.putInt(16); buffer.putShort(1); buffer.putShort(1)
        buffer.putInt(sampleRate); buffer.putInt(sampleRate * 2); buffer.putShort(2); buffer.putShort(16)
        buffer.put("data".toByteArray()); buffer.putInt(dataSize)
        for (sample in audio) buffer.putShort((sample.coerceIn(-1.0f, 1.0f) * 32767).toInt().toShort())
        out.write(buffer.array()); out.flush()
    }
    fun saveWav(audio: FloatArray, file: File, sampleRate: Int) {
        FileOutputStream(file).use { saveWavToStream(audio, it, sampleRate) }
    }
}
