
package com.kittentts.pico

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Bundle
import android.os.Environment
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
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
        requestPermissions()
        setupListeners()
        
        CoroutineScope(Dispatchers.IO).launch {
            try {
                ttsEngine.initialize()
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "TTS Engine Ready", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Init error: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }
    
    private fun requestPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE), 1)
        }
    }
    
    private fun setupListeners() {
        binding.btnGenerate.setOnClickListener { generateSpeech() }
        binding.btnSave.setOnClickListener { saveAudio() }
        binding.btnPlayPause.setOnClickListener { togglePlayback() }
    }
    
    private fun generateSpeech() {
        val text = binding.editTextInput.text.toString()
        if (text.isEmpty()) { Toast.makeText(this, "Please enter text", Toast.LENGTH_SHORT).show(); return }
        
        binding.progressBar.progress = 0
        binding.btnGenerate.isEnabled = false
        
        CoroutineScope(Dispatchers.Main).launch {
            try {
                withContext(Dispatchers.IO) {
                    generatedAudio = ttsEngine.generate(text) { progress ->
                        runOnUiThread { binding.progressBar.progress = progress }
                    }
                }
                binding.progressBar.progress = 100
                Toast.makeText(this@MainActivity, "Speech generated!", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Toast.makeText(this@MainActivity, "Error: ${e.message}", Toast.LENGTH_LONG).show()
            } finally {
                binding.btnGenerate.isEnabled = true
            }
        }
    }
    
    private fun saveAudio() {
        val audio = generatedAudio ?: run { Toast.makeText(this, "No audio to save", Toast.LENGTH_SHORT).show(); return }
        try {
            val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).apply { mkdirs() }
            val file = File(dir, "kitten_tts_${System.currentTimeMillis()}.wav")
            AudioUtils.saveWav(audio, file, 24000)
            Toast.makeText(this, "Saved to Downloads/${file.name}", Toast.LENGTH_LONG).show()
        } catch (e: Exception) {
            Toast.makeText(this, "Save error: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }
    
    private fun togglePlayback() {
        val audio = generatedAudio ?: run { Toast.makeText(this, "No audio to play", Toast.LENGTH_SHORT).show(); return }
        if (isPlaying) stopPlayback() else startPlayback(audio)
    }
    
    private fun startPlayback(audio: FloatArray) {
        isPlaying = true
        binding.btnPlayPause.text = "Pause"
        
        val sampleRate = 24000
        val bufferSize = AudioTrack.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT)
        
        audioTrack = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
            .setAudioFormat(AudioFormat.Builder().setSampleRate(sampleRate).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).setEncoding(AudioFormat.ENCODING_PCM_FLOAT).build())
            .setBufferSizeInBytes(bufferSize)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
            
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
