// 语料检索：把 decay-rag 的「按块时间戳 + 衰减」检索搬进 dsh-go 原生界面。
// 纯 JS，无 python、无 numpy、无嵌入模型；索引在引擎启动后后台建好。
import fs from 'node:fs/promises'
import fsSync from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { createRequire } from 'node:module'

export const name = 'local-decay-rag'

const PLUGIN_DIR = path.dirname(path.dirname(fileURLToPath(import.meta.url)))
const CORPUS_ROOT = process.env.DSH_RAG_CORPUS || '/sdcard/notes'

// 插件目录下的 config.json 可以覆盖：语料根目录、半衰期、返回条数、额外排除目录。
function loadConfig() {
  let raw = {}
  try {
    raw = JSON.parse(fsSync.readFileSync(path.join(PLUGIN_DIR, 'config.json'), 'utf8'))
  } catch {
    raw = {}
  }
  return {
    corpus: process.env.DSH_RAG_CORPUS || raw.corpus || CORPUS_ROOT,
    halfLife: Number(raw.halfLife) > 0 ? Number(raw.halfLife) : HALF_LIFE,
    mode: raw.mode === 'additive' ? 'additive' : 'gate',
    gate: Number(raw.gate) > 0 && Number(raw.gate) <= 1 ? Number(raw.gate) : GATE_BASE,
    topK: Number(raw.topK) > 0 ? Math.min(20, Number(raw.topK)) : TOP_K,
    skip: Array.isArray(raw.skip) ? raw.skip : [],
    refreshMs: process.env.DSH_RAG_REFRESH_MS !== undefined
      ? Number(process.env.DSH_RAG_REFRESH_MS)
      : (Number(raw.refreshMs) >= 0 ? Number(raw.refreshMs) : 20000),
  }
}
const EXTS = ['.md', '.txt']
// 快照 / 归档 / 检索工具的代码目录不进索引：它们只是同一份文档的旧副本或工具说明，
// 会凭空抢走第一命中（实测某个词的命中会被工具自身的 README 顶掉）。
const SKIP_DIRS = new Set([
  '.git', '__pycache__', 'node_modules', 'cache', 'st_snapshot', 'archive', 'rag', 'docker',
  'index', 'index_bge', 'index_qwen', 'index_loc',
  'index_now_bge', 'index_sop', 'index_sop_bge', 'index_sop_loc', 'index_sop_qwen',
])
const SKIP_DIR_RE = /快照|定稿|备份/
const DAY = 86400
const CHUNK_SIZE = 700
const CHUNK_OVERLAP = 120
const W_COS = 0.72
const W_LEX = 0.28
const W_SIM = 0.75
const W_TIME = 0.25
const HALF_LIFE = 30
const GATE_BASE = 0.85
const TOP_K = 6
const COPY_LIMIT = 4000

const CFG = loadConfig()

const RE_DATE = /(20\d{2})-(\d{1,2})-(\d{1,2})/g
const RE_MD = /(?:^|[^\d])(\d{1,2})-(\d{1,2})(?:\s+|·|$)/

// ---------------------------------------------------------------- 核心算法（decay-rag 移植）
function ngrams(text, ns = [2, 3]) {
  const t = String(text).replace(/\s+/g, '').toLowerCase()
  const out = new Map()
  for (const n of ns) {
    for (let i = 0; i + n <= t.length; i++) {
      const g = t.slice(i, i + n)
      out.set(g, (out.get(g) || 0) + 1)
    }
  }
  return out
}

function cosine(a, b) {
  if (!a.size || !b.size) return 0
  let dot = 0
  const [small, big] = a.size <= b.size ? [a, b] : [b, a]
  for (const [g, v] of small) {
    const w = big.get(g)
    if (w) dot += v * w
  }
  let na = 0
  let nb = 0
  for (const v of a.values()) na += v * v
  for (const v of b.values()) nb += v * v
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12)
}

function overlap(q, v) {
  if (!q.size) return 0
  let hit = 0
  for (const g of q.keys()) if (v.has(g)) hit++
  return hit / q.size
}

function splitChunks(text) {
  const paras = String(text).split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean)
  const out = []
  let buf = ''
  for (let p of paras) {
    if (buf.length + p.length + 1 <= CHUNK_SIZE) {
      buf = (buf ? buf + '\n' : '') + p
      continue
    }
    if (buf) out.push(buf)
    while (p.length > CHUNK_SIZE) {
      out.push(p.slice(0, CHUNK_SIZE))
      p = p.slice(CHUNK_SIZE - CHUNK_OVERLAP)
    }
    buf = p
  }
  if (buf) out.push(buf)
  return out
}

function localMidnight(y, mo, d) {
  const t = new Date(y, mo - 1, d, 0, 0, 0, 0).getTime() / 1000
  return Number.isFinite(t) ? t : null
}

function datesIn(text) {
  const res = []
  RE_DATE.lastIndex = 0
  let m
  while ((m = RE_DATE.exec(text))) {
    const ts = localMidnight(Number(m[1]), Number(m[2]), Number(m[3]))
    if (ts !== null) res.push(ts)
  }
  return res
}

// 块级时间戳：文件名日期 → 块内离 mtime 最近的日期 → 10-31 简写 → mtime
function parseChunkTs(text, file, mtime) {
  const head = text.slice(0, 400)
  const cands = datesIn(path.basename(file)).concat(datesIn(head))
  let best = null
  let bestGap = Infinity
  for (const t of cands) {
    if (t > mtime + 2 * DAY) continue
    const gap = Math.abs(mtime - t)
    if (gap < bestGap) {
      best = t
      bestGap = gap
    }
  }
  if (best !== null) return best

  const m = RE_MD.exec(text.slice(0, 300))
  if (m) {
    const mo = Number(m[1])
    const d = Number(m[2])
    if (mo >= 1 && mo <= 12 && d >= 1 && d <= 31) {
      const year = new Date().getFullYear()
      for (const y of [year, year - 1]) {
        const cand = localMidnight(y, mo, d)
        if (cand !== null && cand <= Date.now() / 1000 + 30 * DAY) return cand
      }
    }
  }
  return mtime
}

async function walk(root, out = []) {
  let entries
  try {
    entries = await fs.readdir(root, { withFileTypes: true })
  } catch {
    return out
  }
  for (const e of entries) {
    const p = path.join(root, e.name)
    if (e.isDirectory()) {
      if (SKIP_DIRS.has(e.name) || SKIP_DIR_RE.test(e.name) || CFG.skip.includes(e.name) || e.name.startsWith('.')) continue
      await walk(p, out)
    } else if (e.isFile()) {
      const lower = e.name.toLowerCase()
      if (lower.startsWith('_') || lower.startsWith('.')) continue
      if (!EXTS.some((x) => lower.endsWith(x))) continue
      out.push(p)
    }
  }
  return out
}

async function buildIndex() {
  const files = await walk(CFG.corpus)
  let stamp = ''
  {
    let newest = 0
    let bytes = 0
    for (const f of files) {
      try {
        const st = await fs.stat(f)
        bytes += st.size
        if (st.mtimeMs > newest) newest = st.mtimeMs
      } catch {}
    }
    stamp = `${files.length}:${bytes}:${Math.round(newest)}`
  }
  const docs = []
  for (const file of files) {
    let text
    let mtime
    try {
      text = await fs.readFile(file, 'utf8')
      mtime = (await fs.stat(file)).mtimeMs / 1000
    } catch {
      continue
    }
    const rel = path.relative(CFG.corpus, file)
    splitChunks(text).forEach((chunk, i) => {
      // 文件名也计入打分：否则「因子库总览」这种问法命不中，因为正文里不一定重复标题。
      docs.push({ path: rel, chunk: i, ts: parseChunkTs(chunk, file, mtime), text: chunk, vec: ngrams(rel + '\n' + chunk) })
    })
  }
  return { docs, stamp }
}

async function corpusStamp() {
  const files = await walk(CFG.corpus)
  let newest = 0
  let bytes = 0
  for (const f of files) {
    try {
      const st = await fs.stat(f)
      bytes += st.size
      if (st.mtimeMs > newest) newest = st.mtimeMs
    } catch {}
  }
  return `${files.length}:${bytes}:${Math.round(newest)}`
}

function search(docs, query, k = CFG.topK) {
  const now = Date.now() / 1000
  const q = ngrams(query)
  const rows = docs.map((d) => {
    const cos = cosine(q, d.vec)
    const lex = overlap(q, d.vec)
    const sim = W_COS * cos + W_LEX * lex
    const age = Math.max(0, (now - d.ts) / DAY)
    const decay = Math.pow(0.5, age / CFG.halfLife)
    return { ...d, sim, decay, score: W_SIM * sim + W_TIME * decay }
  })
  rows.sort((a, b) => b.score - a.score)
  const seen = new Set()
  const uniq = []
  for (const r of rows) {
    if (seen.has(r.path)) continue
    seen.add(r.path)
    uniq.push(r)
    if (uniq.length >= k) break
  }
  return uniq
}

function fmtDate(ts) {
  const d = new Date(ts * 1000)
  const p = (n) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`
}

// ---------------------------------------------------------------- 状态
let docs = null
let loading = null
let stamp = ''
let lastQuery = ''
let lastHits = []

function warm() {
  if (docs) return Promise.resolve(docs)
  if (loading) return loading
  loading = buildIndex()
    .then((r) => {
      docs = r.docs
      stamp = r.stamp
      return docs
    })
    .catch(() => null)
    .finally(() => {
      loading = null
    })
  return loading
}


// 插件目录没有 node_modules，宿主包要从引擎安装目录里解析。
async function loadHostPackage(spec) {
  const anchors = [process.argv[1], '/opt/dsh/node_modules/@deepseek-ai/dsh/lib/bin.js']
    .filter((x) => typeof x === 'string' && x)
  for (const anchor of anchors) {
    try {
      return await import(createRequire(anchor).resolve(spec))
    } catch { /* 换下一个锚点 */ }
  }
  return null
}

function renderHits(hits, query) {
  if (!hits.length) return `语料里没有和「${query}」相近的片段。`
  const out = hits.map((h, i) =>
    `${i + 1}. ${h.path} · 第${h.chunk}块 · ${fmtDate(h.ts)} · 综合 ${h.score.toFixed(3)}\n` +
    h.text.replace(/\s+/g, ' ').slice(0, 400))
  return `「${query}」命中 ${hits.length} 块（综合 = 相似 × (0.85 + 0.15×衰减)）：\n\n` + out.join('\n\n')
}

// ---------------------------------------------------------------- 插件
export function apply(ctx) {
  ctx.inject(['dshGo'], (ctx) => {
    setTimeout(warm, 0)

    // 手机上的语料是电脑导出过来的：每次导出 mtime 都会变。
    // 与其让用户记得点「重建索引」，不如自己看目录指纹，变了就静默重建。
    if (CFG.refreshMs > 0) {
      const timer = setInterval(() => {
        if (!docs || loading) return
        corpusStamp().then((s) => {
          if (docs && s !== stamp) {
            docs = null
            warm()
          }
        }).catch(() => {})
      }, CFG.refreshMs)
      if (typeof timer.unref === 'function') timer.unref()
    }

    ctx.dshGo.on('search', (scope, event) => {
      const zh = !String(event.locale ?? '').startsWith('en')
      const q = String(scope.draft ?? '').trim()
      if (!q) return { toast: zh ? '先在输入框写下要查的问题' : 'Type a question first' }
      if (!docs) {
        warm()
        return { toast: zh ? '正在建索引，几秒后再点一次' : 'Indexing, tap once more shortly' }
      }
      lastQuery = q
      lastHits = search(docs, q)
      if (!lastHits.length) return { toast: zh ? '语料里没有相近的片段' : 'No matching passage' }
      return { open: 'panel:hits' }
    })

    ctx.dshGo.panel('hits', {
      view: (scope, caps) => {
        const zh = !String(caps?.locale ?? '').startsWith('en')
        if (!lastHits.length) {
          return {
            type: 'stack',
            children: [
              { type: 'text', text: zh ? '还没有检索过' : 'No search yet' },
              { type: 'text', text: zh ? '在输入框写下问题，点上方「查语料」。' : 'Type a question, then tap Recall.', tone: 'muted' },
            ],
          }
        }
        const top = lastHits[0]
        const snippet = top.text.replace(/\s+/g, ' ').slice(0, 160)
        return {
          type: 'stack',
          title: `「${lastQuery}」`,
          children: [
            { type: 'text', text: zh ? `${lastHits.length} 块命中 · 综合 = 0.75×相似 + 0.25×衰减` : `${lastHits.length} hits · 0.75×sim + 0.25×decay`, tone: 'muted' },
            {
              type: 'list',
              items: lastHits.map((h, i) => ({
                title: `${h.path.replace(/\.md$/i, '')} · 第${h.chunk}块`,
                subtitle: `${fmtDate(h.ts)} · ${h.score.toFixed(3)}`,
                trailing: h.sim.toFixed(2),
                action: { id: 'copy', args: { i } },
              })),
            },
            { type: 'text', text: snippet, tone: 'muted' },
            {
              type: 'actions',
              items: [
                { label: zh ? '复制第一条' : 'Copy top', action: 'top', style: 'primary' },
                { label: zh ? '重查' : 'Re-run', action: 'again' },
                { label: zh ? '重建索引' : 'Rebuild', action: 'rebuild' },
              ],
            },
          ],
        }
      },
      actions: {
        copy: (scope, event) => {
          const i = Number(event?.value?.i ?? 0)
          const h = lastHits[i]
          if (!h) return { toast: '这条已经不在了，重新查一次' }
          return { copy: `${h.path} · ${fmtDate(h.ts)}\n\n${h.text}`.slice(0, COPY_LIMIT), toast: '已复制片段' }
        },
        top: () => {
          const h = lastHits[0]
          if (!h) return { toast: '先查一次' }
          return { copy: `${h.path} · ${fmtDate(h.ts)}\n\n${h.text}`.slice(0, COPY_LIMIT), toast: '已复制第一条' }
        },
        rebuild: () => {
          docs = null
          stamp = ''
          lastHits = []
          warm()
          return { refresh: true, toast: '正在重读语料' }
        },
        again: () => {
          if (!docs || !lastQuery) return { refresh: true }
          lastHits = search(docs, lastQuery)
          return { refresh: true, toast: '已重查' }
        },
      },
    })

  })

  // 模型工具：用户问「我以前写过什么/当时怎么定的」时，模型自己去翻，不用人点胶囊。
  ctx.inject(['tools'], async (ctx) => {
      const pkg = await loadHostPackage('@deepseek-ai/dsh-tools')
      const defineTool = pkg?.defineTool
      if (typeof defineTool !== 'function') return
      ctx.tools.register(defineTool({
        name: 'recall_corpus',
        description: '在本机个人语料（日记、项目笔记、复盘文档）里检索用户早先写过的东西。'
          + '当用户说「我之前写过/上次那个结论/当时怎么定的/翻一下我的笔记」，'
          + '或需要引用用户自己的资料时，先调用本工具；返回「没有找到」再说明没有。'
          + 'query 只放关键词，不要整句照抄。结果按相似度乘以时间衰减排序，越新越靠前。',
        parameters: {
          query: { type: 'string', required: true, description: '要查的关键词或事物名，例如「部署流程」「周报 归档」。' },
          limit: { type: 'integer', description: '最多返回几条，默认 5，最大 10。' },
        },
        output: {
          schema: { type: 'string' },
          render: (_args, value) => [{ type: 'text', text: String(value) }],
        },
        timeoutMs: 20000,
        isConcurrencySafe: () => true,
        execute: async (args) => {
          const query = String(args?.query ?? '').trim()
          const limit = Math.min(Math.max(Number(args?.limit) || 5, 1), 10)
          if (query === '') return 'query 不能为空：请给出要查的关键词。'
          const d = await warm()
          if (!d) return '语料索引还没建好，或者语料目录读不到。'
          return renderHits(search(d, query, limit), query)
        },
      }))
    })
}
