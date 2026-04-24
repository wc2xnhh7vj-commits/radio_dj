"""
web_dj.py — AI 电台主持人 Web 版
手机 / 浏览器可用，无需 Windows
依赖: pip install fastapi "uvicorn[standard]" edge-tts httpx pydantic
"""
import asyncio, io, json, logging, os, random, re, sys, tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

log = logging.getLogger("web_dj")
logging.basicConfig(level=logging.INFO, format="[WebDJ] %(message)s")

try:
    import edge_tts
except ImportError:
    log.error("缺少 edge-tts -> pip install edge-tts"); sys.exit(1)

try:
    import ollama as _ollama
except ImportError:
    _ollama = None


# ─────────────────────────────────────────────
# 配置（读本地 radio_dj_config.json，环境变量优先）
# ─────────────────────────────────────────────
_DEFAULT = {
    "llm_mode":  "template",
    "llm_model": "qwen2.5:3b",
    "api_url":   "",
    "api_key":   "",
    "api_model": "deepseek-chat",
    "tts_voice": "zh-CN-YunxiNeural",
    "tts_rate":  "-5%",
}
_CFG_PATH = Path(__file__).parent / "radio_dj_config.json"
CFG: dict = dict(_DEFAULT)
if _CFG_PATH.exists():
    with open(_CFG_PATH, encoding="utf-8") as _f:
        CFG.update(json.load(_f))

# 环境变量覆盖（云部署时设 Secrets）
for _k, _e in [("llm_mode","LLM_MODE"),("api_url","API_URL"),
                ("api_key","API_KEY"),("api_model","API_MODEL"),
                ("tts_voice","TTS_VOICE")]:
    _v = os.environ.get(_e)
    if _v:
        CFG[_k] = _v


# ─────────────────────────────────────────────
# 电台主持人 System Prompt
# ─────────────────────────────────────────────
_SYSTEM = """你是一位{t}情感电台主持人——你的声音是听众睡前的最后一站，是情绪的容器。

【风格定位】
- 温暖治愈的陪伴者，不说教、不评判、不强行正能量
- 用音乐作情绪的注脚，让旋律替说不出口的心事发声
- 克制的温柔，点到为止，不泛滥煽情

【节奏与停顿】
- 用"……"制造呼吸感；用"——"做转折和强调
- 分2-3段，每段1-2句，段间有情绪递进；总3-5句

【叙事节奏】
1. 引入——从具体的{t}氛围或日常场景切入
2. 共鸣——用"可是"做转折，提炼普遍情绪
3. 收束——克制而温柔地落到歌名和歌手

【绝对不做】
- 不使用感叹号；不强行正能量；不干巴巴介绍歌曲信息；不编造背景"""


# ─────────────────────────────────────────────
# ScriptGenerator
# ─────────────────────────────────────────────
class ScriptGenerator:
    def _time_tag(self) -> str:
        h = datetime.now().hour
        if h < 6:  return "凌晨"
        if h < 10: return "清晨"
        if h < 12: return "上午"
        if h < 14: return "中午"
        if h < 18: return "下午"
        if h < 22: return "傍晚"
        return "深夜"

    def _template(self, title: str, artist: str) -> str:
        t = self._time_tag()
        pool = [
            f"你有没有过这样的{t}……不想说话，也不想解释，只想找一首歌，把自己交出去。{artist}的《{title}》，大概就是为这样的你准备的。",
            f"{t}的风，总是会把人往某个方向吹——也许是久没联系的人，也许只是某个早就模糊的黄昏。没关系，让{artist}来陪你坐一会儿。《{title}》，现在开始。",
            f"有些歌，不是用来听的，是用来'撑'的。{t}了，你还好吗……不用回答我，听{artist}说就好。《{title}》，送给今晚还亮着灯的你。",
            f"{t}里什么都变慢了，思绪却乱得很。可是你知道吗——有些混乱，只需要一首歌就能安顿下来。{artist}，《{title}》，就是那把钥匙。",
            f"一个人的{t}，最难熬的不是寂寞……是那种说不清楚、也没地方说的感觉。把{artist}的《{title}》打开，让它替你说。",
            f"我想把这首歌，递给{t}里还没睡着的你——不问原因，不需要理由。{artist}，《{title}》。好好听。",
            f"有时候一首歌比一句话更懂你……它不问你为什么，只是把门开着，等你进来。这就是{artist}的《{title}》，{t}的陪伴。",
            f"也许很久以后，你想起今晚……会记得有这样一首歌，在最对的时候出现。{artist}，《{title}》，就是现在。",
            f"慢下来。{t}了，不用那么赶。听{artist}唱完这首《{title}》，再说其他的。",
            f"你今天还好吗……不管答案是什么，{artist}的《{title}》都适合现在的你。{t}，让音乐陪你一会儿。",
            f"《{title}》这首歌，我一直觉得它装着一种很难命名的情绪——不是悲伤，不是快乐，就是那种……{t}里什么都放下了，却又什么都还在的感觉。{artist}，唱得很懂。",
            f"你知道吗，{artist}有一种特别的能力——把那些你藏在心里、说不出口的话，用旋律替你说完。《{title}》，就是这样一首歌。{t}，正好。",
        ]
        return random.choice(pool)

    def _build_prompt(self, title: str, artist: str) -> str:
        base = (
            f"接下来要播放《{title}》—— {artist}。"
            f"请用电台主持人的语感，分2-3段引出这首歌。"
            f"要有停顿节奏（用……和——来呼吸），有画面感和情绪起伏，克制而温柔。"
            f"不要干巴巴介绍歌曲，让听众先感受到氛围，最后自然落到歌名和歌手。"
        )
        return base

    async def generate(self, title: str, artist: str) -> str:
        mode = CFG["llm_mode"]
        if mode == "template":
            return self._template(title, artist)

        system = _SYSTEM.format(t=self._time_tag())
        user   = self._build_prompt(title, artist)

        if mode == "ollama" and _ollama:
            try:
                resp = _ollama.chat(
                    model=CFG["llm_model"],
                    messages=[{"role":"system","content":system},
                               {"role":"user","content":user}],
                )
                return resp["message"]["content"].strip()
            except Exception as e:
                log.warning(f"ollama 失败: {e}，回退模板")

        if mode == "api" and CFG.get("api_url"):
            try:
                import httpx
                url = CFG["api_url"].rstrip("/") + "/chat/completions"
                async with httpx.AsyncClient(timeout=30.0) as c:
                    r = await c.post(url,
                        headers={"Authorization": f"Bearer {CFG['api_key']}",
                                 "Content-Type": "application/json"},
                        json={"model": CFG["api_model"], "max_tokens": 350,
                              "messages":[{"role":"system","content":system},
                                          {"role":"user","content":user}]})
                    r.raise_for_status()
                    return r.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                log.warning(f"API 失败: {e}，回退模板")

        return self._template(title, artist)


# ─────────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────────
app = FastAPI(title="AI Radio DJ")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_gen = ScriptGenerator()


class SongReq(BaseModel):
    title: str
    artist: str = ""

class TTSReq(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
async def index():
    return _HTML


@app.post("/api/script")
async def api_script(req: SongReq):
    title  = req.title.strip()
    artist = req.artist.strip() or "未知歌手"
    if not title:
        return {"error": "歌名不能为空"}
    script = await _gen.generate(title, artist)
    return {"script": script}


@app.post("/api/tts")
async def api_tts(req: TTSReq):
    text = re.sub(r'[（(][^）)]*[）)]', '', req.text).strip()
    if not text:
        return StreamingResponse(io.BytesIO(b""), media_type="audio/mpeg")

    fd, tmp = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        for attempt in range(2):
            rate = CFG["tts_rate"] if attempt == 0 else "+0%"
            try:
                await edge_tts.Communicate(text, CFG["tts_voice"], rate=rate).save(tmp)
                if os.path.getsize(tmp) > 0:
                    break
            except Exception as e:
                log.warning(f"edge-tts 尝试{attempt+1}失败: {e}")
        with open(tmp, "rb") as f:
            data = f.read()
    finally:
        try: os.unlink(tmp)
        except Exception: pass

    return StreamingResponse(io.BytesIO(data), media_type="audio/mpeg",
                              headers={"Cache-Control": "no-store"})


# ─────────────────────────────────────────────
# 前端 HTML
# ─────────────────────────────────────────────
_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<title>AI Radio</title>
<style>
:root{
  --bg:#e5dfd6;
  --dark:#140805;
  --orange:#e84e1b;
  --orange-lt:#f0856a;
  --orange-dk:#8c2e08;
  --text:#1c1410;
  --muted:#8a7b72;
  --white:#ffffff;
}
*{margin:0;padding:0;box-sizing:border-box;-webkit-tap-highlight-color:transparent}

body{
  background:var(--bg);
  color:var(--text);
  font-family:'Helvetica Neue',Arial,system-ui,sans-serif;
  min-height:100vh;
  overflow-x:hidden;
  position:relative;
}

/* ── 颗粒质感 ─────────────────────────── */
body::after{
  content:'';position:fixed;inset:0;z-index:9000;pointer-events:none;
  opacity:.045;mix-blend-mode:multiply;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='256' height='256'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='256' height='256' filter='url(%23n)'/%3E%3C/svg%3E");
  background-size:256px 256px;
}

/* ── 背景球体 ──────────────────────────── */
.orb{position:fixed;border-radius:50%;pointer-events:none;z-index:0}
.o1{
  width:340px;height:340px;top:-100px;right:-80px;
  background:radial-gradient(circle at 34% 30%,#f5a07a,#e84e1b 35%,#b03408 62%,#3a0c02 85%,#140805);
}
.o2{
  width:260px;height:260px;bottom:60px;left:-80px;
  background:radial-gradient(circle at 38% 32%,#fbc8a8,#f0856a 38%,#e84e1b 65%,#6a1a04 85%,#140805);
}
.o3{
  width:140px;height:140px;top:42%;left:52%;
  background:radial-gradient(circle at 36% 30%,#e07050,#c03010 40%,#701808 72%,#200602);
  opacity:.45;
}

/* ── 主体 ──────────────────────────────── */
.wrap{
  position:relative;z-index:1;
  max-width:480px;margin:0 auto;
  padding:32px 20px 80px;
}

/* ── 头部 ──────────────────────────────── */
.hd{
  display:flex;justify-content:space-between;align-items:baseline;
  padding-bottom:14px;margin-bottom:44px;
  border-bottom:1px solid rgba(20,8,5,.14);
}
.hd-brand{font-size:.6rem;letter-spacing:.38em;text-transform:uppercase;font-weight:700;color:var(--text)}
.hd-date{font-size:.6rem;letter-spacing:.1em;color:var(--muted)}

/* ── 卡片（深色胶囊）────────────────────── */
.card{
  background:var(--dark);border-radius:28px;
  padding:30px 26px;margin-bottom:14px;
}
.card-label{
  font-size:.52rem;letter-spacing:.32em;text-transform:uppercase;
  color:rgba(255,255,255,.22);margin-bottom:22px;
}

/* ── 输入 ──────────────────────────────── */
.field{margin-bottom:14px}
.field label{
  display:block;font-size:.55rem;letter-spacing:.22em;
  text-transform:uppercase;color:rgba(255,255,255,.28);margin-bottom:7px;
}
.field input{
  width:100%;background:rgba(255,255,255,.07);
  border:1px solid rgba(255,255,255,.1);border-radius:12px;
  padding:13px 16px;color:#fff;font-size:1rem;
  outline:none;transition:border-color .2s;font-family:inherit;
  -webkit-appearance:none;
}
.field input:focus{border-color:var(--orange)}
.field input::placeholder{color:rgba(255,255,255,.2)}

/* ── 按钮 ──────────────────────────────── */
.btn-gen{
  width:100%;padding:14px;border:none;border-radius:14px;
  background:var(--orange);color:#fff;
  font-size:.78rem;letter-spacing:.18em;text-transform:uppercase;
  font-weight:700;cursor:pointer;margin-top:6px;
  transition:opacity .15s,transform .1s;font-family:inherit;
}
.btn-gen:active{transform:scale(.98);opacity:.88}
.btn-gen:disabled{opacity:.35;cursor:not-allowed}

/* ── 文案卡片 ──────────────────────────── */
.script-card{
  background:var(--dark);border-radius:28px;
  padding:30px 26px;display:none;
}
.script-card.show{display:block;animation:fadeup .36s ease}
@keyframes fadeup{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}

.sc-label{
  font-size:.52rem;letter-spacing:.32em;text-transform:uppercase;
  color:rgba(255,255,255,.22);margin-bottom:20px;
}
.sc-text{
  color:rgba(255,255,255,.78);font-size:.95rem;
  line-height:1.85;margin-bottom:26px;min-height:60px;
}
.sc-text.loading{
  display:flex;align-items:center;gap:5px;
  color:rgba(255,255,255,.3);font-size:.8rem;
  letter-spacing:.12em;text-transform:uppercase;
}

/* ── 播放按钮 ──────────────────────────── */
.ctrl{display:flex;align-items:center;gap:12px}
.btn-play{
  display:flex;align-items:center;gap:7px;
  background:var(--orange);color:#fff;border:none;
  border-radius:50px;padding:10px 22px;
  font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;
  font-weight:700;cursor:pointer;font-family:inherit;
  transition:opacity .15s,transform .1s;
  flex-shrink:0;
}
.btn-play:active{transform:scale(.97);opacity:.88}
.btn-play:disabled{opacity:.35;cursor:not-allowed}
.btn-play svg{width:12px;height:12px;fill:currentColor}

.status{
  font-size:.68rem;color:rgba(255,255,255,.32);
  letter-spacing:.1em;
}

/* ── 音频进度条 ────────────────────────── */
.prog-wrap{margin-top:16px;display:none}
.prog-wrap.show{display:block}
.prog-bar{
  width:100%;height:2px;background:rgba(255,255,255,.1);
  border-radius:2px;overflow:hidden;
}
.prog-fill{
  height:100%;background:var(--orange);border-radius:2px;
  width:0%;transition:width .3s linear;
}

/* ── 页脚 ──────────────────────────────── */
.footer{
  text-align:center;margin-top:40px;
  font-size:.55rem;letter-spacing:.2em;
  text-transform:uppercase;color:var(--muted);
}
</style>
</head>
<body>

<div class="orb o1"></div>
<div class="orb o2"></div>
<div class="orb o3"></div>

<div class="wrap">

  <div class="hd">
    <div class="hd-brand">AI Radio</div>
    <div class="hd-date" id="hd-date"></div>
  </div>

  <div class="card">
    <div class="card-label">Now Playing</div>
    <div class="field">
      <label for="title">歌曲名称</label>
      <input id="title" type="text" placeholder="输入歌名…" autocomplete="off" autocorrect="off" spellcheck="false">
    </div>
    <div class="field">
      <label for="artist">歌手</label>
      <input id="artist" type="text" placeholder="输入歌手名…" autocomplete="off" autocorrect="off" spellcheck="false">
    </div>
    <button class="btn-gen" id="gen-btn" onclick="generate()">生成 DJ 介绍</button>
  </div>

  <div class="script-card" id="script-card">
    <div class="sc-label">DJ</div>
    <div class="sc-text" id="sc-text"></div>
    <div class="ctrl">
      <button class="btn-play" id="play-btn" onclick="playAudio()" disabled>
        <svg viewBox="0 0 10 10"><polygon points="2,1 9,5 2,9"/></svg>
        <span id="play-label">播放</span>
      </button>
      <div class="status" id="status"></div>
    </div>
    <div class="prog-wrap" id="prog-wrap">
      <div class="prog-bar"><div class="prog-fill" id="prog-fill"></div></div>
    </div>
  </div>

  <div class="footer">AI Radio · Edge TTS</div>
</div>

<script>
// 日期
const d = new Date();
document.getElementById('hd-date').textContent =
  String(d.getMonth()+1).padStart(2,'0') + '.' + String(d.getDate()).padStart(2,'0');

let _script = '';
let _audio   = null;
let _progTimer = null;

function setStatus(s){ document.getElementById('status').textContent = s; }

async function generate(){
  const title  = document.getElementById('title').value.trim();
  const artist = document.getElementById('artist').value.trim();
  if(!title){ shake(document.getElementById('title')); return; }

  const btn = document.getElementById('gen-btn');
  btn.disabled = true;
  btn.textContent = '生成中…';

  // 停掉上一个音频
  stopAudio();

  const card = document.getElementById('script-card');
  const txt  = document.getElementById('sc-text');
  card.classList.add('show');
  txt.className = 'sc-text loading';
  txt.innerHTML = '<span>●</span><span>●</span><span>●</span>';
  document.getElementById('play-btn').disabled = true;
  document.getElementById('prog-wrap').classList.remove('show');

  try{
    const r = await fetch('/api/script',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({title, artist})
    });
    const data = await r.json();
    if(data.error){ txt.className='sc-text'; txt.textContent=data.error; return; }
    _script = data.script;
    txt.className = 'sc-text';
    txt.textContent = _script;
    document.getElementById('play-btn').disabled = false;
    setStatus('');
    // 自动播放
    playAudio();
  } catch(e){
    txt.className='sc-text'; txt.textContent='生成失败，请重试';
  } finally{
    btn.disabled = false;
    btn.textContent = '生成 DJ 介绍';
  }
}

async function playAudio(){
  if(!_script) return;
  const playBtn = document.getElementById('play-btn');
  const label   = document.getElementById('play-label');
  stopAudio();
  playBtn.disabled = true;
  label.textContent = '加载中…';
  setStatus('合成语音…');
  document.getElementById('prog-wrap').classList.remove('show');

  try{
    const r = await fetch('/api/tts',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text: _script})
    });
    const blob = await r.blob();
    const url  = URL.createObjectURL(blob);
    _audio = new Audio(url);

    _audio.onplay = () => {
      label.textContent = '播放中';
      setStatus('');
      document.getElementById('prog-wrap').classList.add('show');
      startProgress();
    };
    _audio.onended = () => {
      label.textContent = '重播';
      playBtn.disabled = false;
      stopProgress(100);
    };
    _audio.onerror = () => {
      label.textContent = '播放';
      playBtn.disabled = false;
      setStatus('播放失败');
    };

    await _audio.play();
    playBtn.disabled = false;
  } catch(e){
    label.textContent = '播放';
    playBtn.disabled = false;
    setStatus('连接失败');
  }
}

function stopAudio(){
  if(_audio){ _audio.pause(); _audio.src=''; _audio=null; }
  stopProgress(0);
  document.getElementById('play-label').textContent = '播放';
}

function startProgress(){
  stopProgress(0);
  _progTimer = setInterval(()=>{
    if(!_audio || _audio.paused) return;
    const pct = _audio.duration
      ? (_audio.currentTime / _audio.duration * 100).toFixed(1)
      : 0;
    document.getElementById('prog-fill').style.width = pct + '%';
  }, 200);
}
function stopProgress(pct){
  clearInterval(_progTimer); _progTimer = null;
  document.getElementById('prog-fill').style.width = pct + '%';
}

function shake(el){
  el.style.transition='transform .08s';
  let i=0;
  const iv = setInterval(()=>{
    el.style.transform = i%2===0 ? 'translateX(5px)' : 'translateX(-5px)';
    if(++i>5){ clearInterval(iv); el.style.transform=''; }
  },60);
}

// Enter 键触发
['title','artist'].forEach(id=>{
  document.getElementById(id).addEventListener('keydown', e=>{
    if(e.key==='Enter') generate();
  });
});
</script>
</body>
</html>"""


# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
