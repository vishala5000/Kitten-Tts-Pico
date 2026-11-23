package com.kittentts.pico

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.zip.ZipInputStream
import kotlin.math.min
import kotlin.math.roundToInt

class KittenTtsEngine(private val context: Context) {
    data class Voice(val key: String, val name: String, val lang: String, val description: String)
    private val env = OrtEnvironment.getEnvironment()
    private val sessions = HashMap<String, OrtSession>()
    private val packs = HashMap<String, HashMap<String, FloatArray>>()
    private var packRows = 510
    private var packCols = 256
    private val vocab = HashMap<String, Int>()
    private val lexicon = HashMap<String, String>()
    var sampleRate = 24000; private set
    val voices = ArrayList<Voice>()

    private fun readAsset(name: String): String =
        context.assets.open(name).bufferedReader().use { it.readText() }

    fun initialize() {
        val cfg = JSONObject(readAsset("config.json"))
        sampleRate = cfg.optInt("sample_rate", 24000)
        val vs = cfg.getJSONObject("voices")
        for (lang in listOf("en", "hi")) {
            val arr = vs.optJSONArray(lang) ?: continue
            for (i in 0 until arr.length()) {
                val o = arr.getJSONObject(i)
                voices.add(Voice(o.getString("key"), o.getString("name"), lang, o.optString("description", "")))
            }
        }
        val vj = JSONObject(readAsset("vocab.json"))
        val vk = vj.keys(); while (vk.hasNext()) { val k = vk.next(); vocab[k] = vj.getInt(k) }
        val lj = JSONObject(readAsset("cmu_misaki_us.json"))
        val lk = lj.keys(); while (lk.hasNext()) { val k = lk.next(); lexicon[k] = lj.optString(k, "") }
        packs["en"] = loadPack("voices_en.npz")
        packs["hi"] = loadPack("voices_hi.npz")
    }

    private fun loadPack(asset: String): HashMap<String, FloatArray> {
        val out = HashMap<String, FloatArray>()
        ZipInputStream(context.assets.open(asset)).use { zis ->
            var e = zis.nextEntry
            while (e != null) {
                if (e.name.endsWith(".npy")) {
                    val b = zis.readBytes()
                    val major = b[6].toInt()
                    val hl: Int; val ds: Int
                    if (major == 1) { hl = (b[8].toInt() and 0xFF) or ((b[9].toInt() and 0xFF) shl 8); ds = 10 + hl }
                    else { hl = (b[8].toInt() and 0xFF) or ((b[9].toInt() and 0xFF) shl 8) or ((b[10].toInt() and 0xFF) shl 16) or ((b[11].toInt() and 0xFF) shl 24); ds = 12 + hl }
                    val header = String(b, ds - hl, hl, Charsets.US_ASCII)
                    val si = header.indexOf("'shape'")
                    if (si >= 0) {
                        val p1 = header.indexOf('(', si); val p2 = header.indexOf(')', p1)
                        if (p1 >= 0 && p2 > p1) {
                            val nums = ArrayList<Int>()
                            for (p in header.substring(p1 + 1, p2).split(",")) { val n = p.trim().toIntOrNull(); if (n != null) nums.add(n) }
                            if (nums.size >= 2) { packRows = nums[0]; packCols = nums[1] }
                        }
                    }
                    val count = (b.size - ds) / 4
                    val f = FloatArray(count)
                    val bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN)
                    for (i in 0 until count) f[i] = bb.getFloat(ds + i * 4)
                    out[e.name.removeSuffix(".npy")] = f
                }
                e = zis.nextEntry
            }
        }
        return out
    }

    private fun session(lang: String): OrtSession {
        var s = sessions[lang]
        if (s == null) {
            val file = if (lang == "hi") "kitten_tts_2_pico_hi_int4.onnx" else "kitten_tts_2_pico_en_int4.onnx"
            val so = OrtSession.SessionOptions(); so.setIntraOpNumThreads(2)
            s = env.createSession(context.assets.open(file).readBytes(), so)
            sessions[lang] = s
        }
        return s
    }

    private fun tokens(ps: String): LongArray {
        val ids = ArrayList<Long>(); ids.add(0L)
        for (ch in ps) {
            val id = vocab[ch.toString()]
            if (id != null) ids.add(id.toLong())
            else if (ch == '(' || ch == ')') { val c = vocab[","]; if (c != null) ids.add(c.toLong()) }
        }
        ids.add(10L); ids.add(0L)
        return ids.toLongArray()
    }

    private fun synthChunk(lang: String, key: String, ps: String, speed: Float, keepStart: Boolean): FloatArray {
        val tok = tokens(ps)
        val pack = packs[lang]?.get(key) ?: throw RuntimeException("voice $key not loaded")
        val idx = min(tok.size - 1, packRows - 1)
        val style = FloatArray(packCols)
        System.arraycopy(pack, idx * packCols, style, 0, packCols)
        val s = session(lang)
        val idT = OnnxTensor.createTensor(env, LongBuffer.wrap(tok), longArrayOf(1, tok.size.toLong()))
        val stT = OnnxTensor.createTensor(env, FloatBuffer.wrap(style), longArrayOf(1, packCols.toLong()))
        val spT = OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArrayOf(speed)), longArrayOf(1))
        val res = s.run(mapOf("input_ids" to idT, "style" to stT, "speed" to spT))
        val wb = (res.get(0).value as OnnxTensor).floatBuffer
        val wav = FloatArray(wb.remaining()); wb.get(wav)
        val db = (res.get(1).value as OnnxTensor).longBuffer
        val dur = LongArray(db.remaining()); db.get(dur)
        idT.close(); stT.close(); spT.close(); res.close()
        for (i in wav.indices) wav[i] = wav[i].coerceIn(-1f, 1f)
        var total = 0L; for (d in dur) total += d
        val spf = maxOf(1, (wav.size.toDouble() / maxOf(total, 1L).toDouble()).roundToInt())
        val lead = if (keepStart) 0 else maxOf(0, (dur[0] - 1).toInt() * spf)
        val trail = maxOf(0, (dur[dur.size - 1] - 1).toInt() * spf)
        var w = wav
        if (lead + trail + spf < wav.size) {
            val end = if (trail > 0) wav.size - trail else wav.size
            w = wav.copyOfRange(lead, end)
        }
        val f = min(120, w.size / 8)
        if (f > 0) for (i in 0 until f) {
            val g = i.toFloat() / f
            w[i] = w[i] * g; w[w.size - 1 - i] = w[w.size - 1 - i] * g
        }
        return w
    }

    fun generate(text: String, voice: Voice, speed: Float, progress: (Int) -> Unit): FloatArray {
        val chunks = chunkText(text, voice.lang)
        if (chunks.isEmpty()) return FloatArray(0)
        val gap = FloatArray((0.15f * sampleRate).toInt())
        val parts = ArrayList<FloatArray>()
        for (i in chunks.indices) {
            parts.add(synthChunk(voice.lang, voice.key, chunks[i], speed, i == 0))
            if (i < chunks.size - 1) parts.add(gap)
            progress(((i + 1) * 100) / chunks.size)
        }
        var n = 0; for (p in parts) n += p.size
        val out = FloatArray(n); var off = 0
        for (p in parts) { System.arraycopy(p, 0, out, off, p.size); off += p.size }
        return out
    }

    private fun chunkText(text: String, lang: String): List<String> {
        val whole = if (lang == "hi") phonemizeHi(text) else phonemizeEn(text)
        if (whole.isEmpty()) return emptyList()
        if (whole.length <= 350) return listOf(whole)
        val chunks = ArrayList<String>(); var cur = StringBuilder()
        for (sent in splitSentences(text)) {
            val ps = if (lang == "hi") phonemizeHi(sent) else phonemizeEn(sent)
            for (piece in wrap(ps)) {
                if (cur.isNotEmpty() && cur.length + 1 + piece.length > 350) { chunks.add(cur.toString()); cur = StringBuilder(piece) }
                else { if (cur.isNotEmpty()) cur.append(' '); cur.append(piece) }
            }
        }
        if (cur.isNotEmpty()) chunks.add(cur.toString())
        return chunks
    }

    private fun splitSentences(text: String): List<String> {
        val out = ArrayList<String>(); val sb = StringBuilder()
        for (i in text.indices) {
            sb.append(text[i])
            if (text[i] in ".!?…" || text[i] == '।') {
                if (i + 1 >= text.length || text[i + 1].isWhitespace()) { out.add(sb.toString()); sb.setLength(0) }
            }
        }
        if (sb.isNotEmpty()) out.add(sb.toString())
        return out
    }

    private fun wrap(ps: String): List<String> {
        var s = ps; val out = ArrayList<String>()
        while (s.length > 350) {
            var cut = s.lastIndexOf(' ', 350); if (cut <= 0) cut = 350
            out.add(s.substring(0, cut)); s = s.substring(cut).trim()
        }
        if (s.isNotEmpty()) out.add(s)
        return out
    }

    private fun filterVocab(ps: String): String {
        val sb = StringBuilder()
        for (c in ps) if (vocab.containsKey(c.toString())) sb.append(c)
        return sb.toString()
    }

    private fun lookupEn(w: String): String {
        lexicon[w]?.let { if (it.isNotEmpty()) return it }
        if (w.endsWith("'s")) {
            val stem = lexicon[w.substring(0, w.length - 2)]
            if (stem != null && stem.isNotEmpty()) {
                val last = stem[stem.length - 1]
                return stem + when { "ptkfθ".indexOf(last) >= 0 -> "s"; "szʃʒʧʤ".indexOf(last) >= 0 -> "ᵻz"; else -> "z" }
            }
        }
        if (w.contains("-")) {
            val parts = w.split("-").map { lookupEn(it) }.filter { it.isNotEmpty() }
            if (parts.isNotEmpty()) return parts.joinToString(" ")
        }
        return letterFallback(w)
    }

    private fun letterFallback(w: String): String {
        val sb = StringBuilder(); var i = 0
        while (i < w.length) {
            val two = if (i + 1 < w.length) w.substring(i, i + 2) else ""
            val d = when (two) { "sh" -> "ʃ"; "ch" -> "ʧ"; "th" -> "θ"; "ph" -> "f"; "ck" -> "k"; "ee" -> "iː"; "oo" -> "uː"; "ea" -> "iː"; "ou" -> "W"; "ow" -> "W"; "ai" -> "A"; "ay" -> "A"; "oa" -> "O"; "oi" -> "Y"; "oy" -> "Y"; "qu" -> "kw"; "ng" -> "ŋ"; else -> null }
            if (d != null) { sb.append(d); i += 2; continue }
            val c = w[i]
            sb.append(when (c) { 'a' -> "æ"; 'e' -> "ɛ"; 'i' -> "ɪ"; 'o' -> "ɑ"; 'u' -> "ʌ"; 'c' -> "k"; 'g' -> "ɡ"; 'j' -> "ʤ"; 'q' -> "k"; 'r' -> "ɹ"; 'x' -> "ks"; 'y' -> "j"; else -> c.toString() })
            i += 1
        }
        return sb.toString()
    }

    private fun phonemizeEn(text: String): String {
        val out = ArrayList<String>(); val word = StringBuilder()
        fun flush() {
            if (word.isEmpty()) return
            val w = word.toString().lowercase(); word.setLength(0)
            val ps = lookupEn(w); if (ps.isNotEmpty()) out.add(ps)
        }
        for (ch in text) {
            when {
                ch.isLetter() || ch == '\'' || ch == '-' -> word.append(ch)
                ch in ".,!?;:" -> { flush(); out.add(ch.toString()) }
                else -> flush()
            }
        }
        flush()
        return filterVocab(out.joinToString(" ").replace("ɾ", "T").replace("ʔ", "t"))
    }

    private val hiC = mapOf('क' to "k",'ख' to "kʰ",'ग' to "ɡ",'घ' to "ɡʱ",'ङ' to "ŋ",'च' to "ʧ",'छ' to "ʧʰ",'ज' to "ʤ",'झ' to "ʤʱ",'ञ' to "ɲ",'ट' to "ʈ",'ठ' to "ʈʰ",'ड' to "ɖ",'ढ' to "ɖʱ",'ण' to "ɳ",'त' to "t",'थ' to "tʰ",'द' to "d",'ध' to "dʱ",'न' to "n",'प' to "p",'फ' to "pʰ",'ब' to "b",'भ' to "bʱ",'म' to "m",'य' to "j",'र' to "r",'ल' to "l",'व' to "ʋ",'श' to "ʃ",'ष' to "ʂ",'स' to "s",'ह' to "h",'क़' to "q",'ख़' to "x",'ग़' to "ɣ",'ज़' to "z",'फ़' to "f",'ड़' to "ɽ",'ढ़' to "ɽʱ")
    private val hiV = mapOf('अ' to "ə",'आ' to "aː",'इ' to "ɪ",'ई' to "iː",'उ' to "ʊ",'ऊ' to "uː",'ए' to "eː",'ऐ' to "ɛː",'ओ' to "oː",'औ' to "ɔː",'ऑ' to "ɒ")
    private val hiM = mapOf('ा' to "aː",'ि' to "ɪ",'ी' to "iː",'ु' to "ʊ",'ू' to "uː",'े' to "eː",'ै' to "ɛː",'ो' to "oː",'ौ' to "ɔː",'ृ' to "ɪ")

    private fun phonemizeHi(text: String): String {
        val t = text.replace('।', '.')
        val sb = StringBuilder(); var i = 0
        while (i < t.length) {
            val ch = t[i]
            val base = hiC[ch]
            if (base != null) {
                i += 1
                if (i < t.length && t[i] == '़') i += 1
                if (i < t.length && t[i] == '्') { sb.append(base); i += 1 }
                else if (i < t.length && hiM.containsKey(t[i])) { sb.append(base); sb.append(hiM[t[i]]); i += 1 }
                else { sb.append(base); sb.append("ə") }
            } else if (hiV.containsKey(ch)) { sb.append(hiV[ch]); i += 1 }
            else if (ch == 'ं' || ch == 'ँ') { sb.append("̃"); i += 1 }
            else if (ch == 'ः') { sb.append("h"); i += 1 }
            else if (ch in ".,!?;: ") { sb.append(ch); i += 1 }
            else i += 1
        }
        return filterVocab(sb.toString())
    }

    fun close() { for (s in sessions.values) try { s.close() } catch (_: Exception) {} }
}