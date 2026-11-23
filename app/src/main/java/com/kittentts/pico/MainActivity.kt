package com.kittentts.pico

import android.content.ContentValues
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.kittentts.pico.databinding.ActivityMainBinding
import kotlinx.coroutines.*
import java.io.File

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var engine: KittenTtsEngine
    private var audioTrack: AudioTrack? = null
    private var audio: FloatArray? = null
    private var isPlaying = false
    private var playOffset = 0
    private var job: Job? = null
    private var voiceList = ArrayList<KittenTtsEngine.Voice>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        engine = KittenTtsEngine(this)
        binding.btnGenerate.setOnClickListener { generate() }
        binding.btnSave.setOnClickListener { save() }
        binding.btnPlayPause.setOnClickListener { togglePlay() }
        CoroutineScope(Dispatchers.IO).launch {
            try {
                engine.initialize()
                voiceList.addAll(engine.voices)
                withContext(Dispatchers.Main) {
                    val labels = voiceList.map { "${it.name} (${it.lang})" }
                    binding.spinnerVoice.adapter = ArrayAdapter(this@MainActivity,
                        android.R.layout.simple_spinner_dropdown_item, labels)
                    toast("Ready: ${voiceList.size} voices")
                }
            } catch (e: Exception) { withContext(Dispatchers.Main) { toast("Init error: ${e.message}") } }
        }
    }

    private fun toast(m: String) = Toast.makeText(this, m, Toast.LENGTH_LONG).show()

    private fun selectedVoice(): KittenTtsEngine.Voice? {
        val pos = binding.spinnerVoice.selectedItemPosition
        return if (pos in voiceList.indices) voiceList[pos] else voiceList.firstOrNull()
    }

    private fun generate() {
        val text = binding.editTextInput.text.toString()
        if (text.isEmpty()) { toast("Enter text"); return }
        val v = selectedVoice() ?: run { toast("Engine not ready"); return }
        stopPlayback(); playOffset = 0
        binding.progressBar.progress = 0; binding.btnGenerate.isEnabled = false
        CoroutineScope(Dispatchers.Main).launch {
            try {
                val a = withContext(Dispatchers.IO) {
                    engine.generate(text, v, 1.0f) { p -> runOnUiThread { binding.progressBar.progress = p } }
                }
                audio = a; binding.progressBar.progress = 100
                toast("Generated ${a.size / engine.sampleRate.toFloat()}s")
            } catch (e: Exception) { toast("Error: ${e.message}") }
            finally { binding.btnGenerate.isEnabled = true }
        }
    }

    private fun save() {
        val a = audio ?: run { toast("Nothing to save"); return }
        val name = "kitten_tts_${System.currentTimeMillis()}.wav"
        try {
            if (Build.VERSION.SDK_INT >= 29) {
                val cv = ContentValues().apply {
                    put(MediaStore.Downloads.DISPLAY_NAME, name)
                    put(MediaStore.Downloads.MIME_TYPE, "audio/wav")
                    put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
                }
                val uri = contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, cv)
                    ?: throw RuntimeException("insert failed")
                contentResolver.openOutputStream(uri)?.use { AudioUtils.saveWavToStream(a, it, engine.sampleRate) }
            } else {
                val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).apply { mkdirs() }
                AudioUtils.saveWav(a, File(dir, name), engine.sampleRate)
            }
            toast("Saved $name")
        } catch (e: Exception) {
            try { val f = File(filesDir, name); AudioUtils.saveWav(a, f, engine.sampleRate); toast("Saved internal ${f.name}") }
            catch (e2: Exception) { toast("Save error: ${e2.message}") }
        }
    }

    private fun togglePlay() {
        val a = audio ?: run { toast("Nothing to play"); return }
        if (isPlaying) stopPlayback() else startPlayback(a)
    }

    private fun startPlayback(a: FloatArray) {
        if (playOffset >= a.size) playOffset = 0
        isPlaying = true; binding.btnPlayPause.text = "Pause"
        val rate = engine.sampleRate
        val buf = AudioTrack.getMinBufferSize(rate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT)
        audioTrack = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
            .setAudioFormat(AudioFormat.Builder().setSampleRate(rate).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).setEncoding(AudioFormat.ENCODING_PCM_FLOAT).build())
            .setBufferSizeInBytes(buf).setTransferMode(AudioTrack.MODE_STREAM).build()
        audioTrack?.play()
        job = CoroutineScope(Dispatchers.IO).launch {
            var off = playOffset
            while (off < a.size && isPlaying) {
                val end = minOf(off + 1024, a.size)
                val w = audioTrack?.write(a, off, end - off, AudioTrack.WRITE_BLOCKING) ?: (end - off)
                off += if (w > 0) w else (end - off)
                playOffset = off
            }
            withContext(Dispatchers.Main) {
                if (off >= a.size) { playOffset = 0; stopPlayback() } else stopPlayback()
                binding.btnPlayPause.text = if (playOffset in 1 until (audio?.size ?: 0)) "Resume" else "Play"
            }
        }
    }

    private fun stopPlayback() {
        isPlaying = false; job?.cancel()
        try { audioTrack?.stop(); audioTrack?.release() } catch (_: Exception) {}
        audioTrack = null
        binding.btnPlayPause.text = if (playOffset in 1 until (audio?.size ?: 0)) "Resume" else "Play"
    }

    override fun onDestroy() { super.onDestroy(); stopPlayback(); engine.close() }
}