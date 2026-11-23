package com.kittentts.pico

import android.content.ContentValues
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.kittentts.pico.databinding.ActivityMainBinding
import kotlinx.coroutines.*
import java.io.File

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var ttsEngine: KittenTtsEngine
    private var audioTrack: AudioTrack? = null
    private var generatedAudio: FloatArray? = null
    private var isPlaying = false
    private var playbackJob: Job? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        ttsEngine = KittenTtsEngine(this)
        binding.btnGenerate.setOnClickListener { generateSpeech() }
        binding.btnSave.setOnClickListener { saveAudio() }
        binding.btnPlayPause.setOnClickListener { togglePlayback() }
        CoroutineScope(Dispatchers.IO).launch {
            try {
                ttsEngine.initialize()
                withContext(Dispatchers.Main) { toast("TTS Engine Ready") }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) { toast("Init error: ${e.message}") }
            }
        }
    }

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_LONG).show()

    private fun generateSpeech() {
        val text = binding.editTextInput.text.toString()
        if (text.isEmpty()) { toast("Please enter text"); return }
        binding.progressBar.progress = 0
        binding.btnGenerate.isEnabled = false
        CoroutineScope(Dispatchers.Main).launch {
            try {
                withContext(Dispatchers.IO) {
                    generatedAudio = ttsEngine.generate(text) { p -> runOnUiThread { binding.progressBar.progress = p } }
                }
                binding.progressBar.progress = 100
                toast("Speech generated!")
            } catch (e: Exception) {
                toast("Error: ${e.message}")
            } finally {
                binding.btnGenerate.isEnabled = true
            }
        }
    }

    private fun saveAudio() {
        val audio = generatedAudio ?: run { toast("No audio to save"); return }
        val name = "kitten_tts_${System.currentTimeMillis()}.wav"
        try {
            if (Build.VERSION.SDK_INT >= 29) {
                val values = ContentValues().apply {
                    put(MediaStore.Downloads.DISPLAY_NAME, name)
                    put(MediaStore.Downloads.MIME_TYPE, "audio/wav")
                    put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
                }
                val uri = contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                    ?: throw RuntimeException("MediaStore insert failed")
                contentResolver.openOutputStream(uri)?.use { AudioUtils.saveWavToStream(audio, it, 24000) }
            } else {
                val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).apply { mkdirs() }
                AudioUtils.saveWav(audio, File(dir, name), 24000)
            }
            toast("Saved: $name")
        } catch (e: Exception) {
            try {
                val f = File(filesDir, name)
                AudioUtils.saveWav(audio, f, 24000)
                toast("Saved to internal: ${f.name}")
            } catch (e2: Exception) { toast("Save error: ${e2.message}") }
        }
    }

    private fun togglePlayback() {
        val audio = generatedAudio ?: run { toast("No audio to play"); return }
        if (isPlaying) stopPlayback() else startPlayback(audio)
    }

    private fun startPlayback(audio: FloatArray) {
        isPlaying = true
        binding.btnPlayPause.text = "Pause"
        val rate = 24000
        val buf = AudioTrack.getMinBufferSize(rate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT)
        audioTrack = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
            .setAudioFormat(AudioFormat.Builder().setSampleRate(rate).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).setEncoding(AudioFormat.ENCODING_PCM_FLOAT).build())
            .setBufferSizeInBytes(buf)
            .setTransferMode(AudioTrack.MODE_STREAM).build()
        audioTrack?.play()
        playbackJob = CoroutineScope(Dispatchers.IO).launch {
            var offset = 0
            while (offset < audio.size && isPlaying) {
                val end = minOf(offset + 1024, audio.size)
                audioTrack?.write(audio, offset, end - offset, AudioTrack.WRITE_BLOCKING)
                offset = end
            }
            withContext(Dispatchers.Main) { stopPlayback() }
        }
    }

    private fun stopPlayback() {
        isPlaying = false
        binding.btnPlayPause.text = "Play"
        playbackJob?.cancel()
        try { audioTrack?.stop(); audioTrack?.release() } catch (_: Exception) {}
        audioTrack = null
    }

    override fun onDestroy() {
        super.onDestroy()
        stopPlayback()
        ttsEngine.close()
    }
}