import express from 'express';
import type { Request, Response } from 'express';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT) || 3000;

export interface ModelAttempt {
  model: string;
  thinkingLevel: ThinkingLevel | null;
  delayMs: number;
  timeoutMs?: number;
}

export const ANALYZE_SERVER_BUDGET_MS = 26_000;

export const MODEL_ATTEMPTS: readonly ModelAttempt[] = [
  { model: 'gemini-4-hydrogen', thinkingLevel: ThinkingLevel.LOW, delayMs: 0, timeoutMs: 11_500 },
  { model: 'gemini-3.1-flash-lite', thinkingLevel: null, delayMs: 250, timeoutMs: 7_500 },
  { model: 'gemini-3.1-flash-lite', thinkingLevel: null, delayMs: 400, timeoutMs: 6_500 },
];

export interface GeminiErrorInfo {
  status: number;
  statusText: string;
  message: string;
}

export function parseGeminiErrorInfo(err: unknown): GeminiErrorInfo {
  const errObj = (err && typeof err === 'object' ? err : {}) as {
    status?: unknown;
    code?: unknown;
    statusText?: unknown;
    message?: unknown;
  };
  let status = Number(errObj.status) || Number(errObj.code) || 500;
  let statusText = typeof errObj.statusText === 'string' ? errObj.statusText : '';
  let message = typeof errObj.message === 'string' ? errObj.message.trim() : String(err || '');

  const jsonStart = message.indexOf('{');
  if (jsonStart !== -1) {
    try {
      const parsed = JSON.parse(message.slice(jsonStart)) as {
        error?: { code?: unknown; message?: unknown; status?: unknown };
      };
      if (parsed?.error) {
        const innerCode = Number(parsed.error.code);
        if (Number.isFinite(innerCode) && innerCode >= 400) status = innerCode;
        if (typeof parsed.error.status === 'string') statusText = parsed.error.status;
        if (typeof parsed.error.message === 'string') message = parsed.error.message;
      }
    } catch {
      // Ham mesajı koru
    }
  }

  return {
    status: status >= 400 && status < 600 ? status : 500,
    statusText,
    message: message || 'Görsel analizi sırasında bir hata oluştu.',
  };
}

export function isTransientGeminiError(err: unknown): boolean {
  const info = parseGeminiErrorInfo(err);
  if ([408, 429, 500, 502, 503, 504].includes(info.status)) return true;
  const combined = `${info.statusText} ${info.message}`;
  return /UNAVAILABLE|RESOURCE_EXHAUSTED|DEADLINE_EXCEEDED|INTERNAL|TIMEOUT|TIMED_OUT|ABORT|high demand|overloaded|temporarily|try again|ECONNRESET|ETIMEDOUT|EAI_AGAIN|fetch failed|empty response/i.test(
    combined
  );
}

export function formatFriendlyGeminiError(err: unknown): { status: number; error: string } {
  const info = parseGeminiErrorInfo(err);
  const combined = `${info.statusText} ${info.message}`;

  if (info.status === 403 || info.status === 401 || /API_KEY_INVALID|PERMISSION_DENIED/i.test(combined)) {
    return {
      status: info.status,
      error: 'Gemini API anahtarı geçersiz veya yetkisiz. Settings > Secrets üzerinden kontrol edin.',
    };
  }
  if (info.status === 429 || /RESOURCE_EXHAUSTED/i.test(combined)) {
    return {
      status: 429,
      error: 'API istek limitine ulaşıldı. Birkaç saniye sonra "Tekrar Dene" ile yeniden tarayın.',
    };
  }
  if (
    info.status === 408 ||
    info.status === 504 ||
    /DEADLINE_EXCEEDED|TIMEOUT|TIMED_OUT|ABORT/i.test(combined)
  ) {
    return {
      status: 504,
      error: 'Görsel analizi 30 saniyelik zaman aşımına ulaştı. "Tekrar Dene" veya "Elle Tamamla" kullanabilirsiniz.',
    };
  }
  if (
    info.status === 503 ||
    /UNAVAILABLE|high demand|overloaded|temporarily|try again/i.test(combined)
  ) {
    return {
      status: 503,
      error: 'Yapay zekâ servisi şu an geçici olarak yoğun. "Tekrar Dene" ile yeniden tarayabilir veya "Elle Tamamla" kullanabilirsiniz.',
    };
  }
  return {
    status: info.status,
    error: `Görsel analiz edilemedi (${info.status}). "Tekrar Dene" veya "Elle Tamamla" seçeneğini kullanın.`,
  };
}

function defaultSleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export interface RunFallbackOptions {
  totalBudgetMs?: number;
  parentSignal?: AbortSignal;
}

export async function runWithModelFallback<T>(
  invokeFn: (attempt: ModelAttempt, signal?: AbortSignal) => Promise<T>,
  attempts: readonly ModelAttempt[] = MODEL_ATTEMPTS,
  sleepFn: (ms: number) => Promise<void> = defaultSleep,
  options: RunFallbackOptions = {}
): Promise<{ result: T; modelUsed: string; attemptCount: number }> {
  let lastError: unknown = null;
  const totalBudgetMs = options.totalBudgetMs ?? ANALYZE_SERVER_BUDGET_MS;
  const startedAt = Date.now();

  for (let i = 0; i < attempts.length; i++) {
    if (options.parentSignal?.aborted) {
      throw Object.assign(new Error('Client aborted request (ABORT)'), { status: 499 });
    }
    const elapsedMs = Date.now() - startedAt;
    const remainingBudgetMs = totalBudgetMs - elapsedMs;
    if (remainingBudgetMs <= 500) {
      throw (
        lastError ||
        Object.assign(new Error('Server analysis deadline exceeded (TIMEOUT)'), { status: 504 })
      );
    }

    const attempt = attempts[i];
    if (attempt.delayMs > 0) {
      const waitMs = Math.min(attempt.delayMs, Math.max(0, remainingBudgetMs - 1000));
      if (waitMs > 0) {
        await sleepFn(waitMs);
      }
    }

    const remainingAfterDelayMs = totalBudgetMs - (Date.now() - startedAt);
    if (remainingAfterDelayMs <= 400) {
      throw (
        lastError ||
        Object.assign(new Error('Server analysis deadline exceeded (TIMEOUT)'), { status: 504 })
      );
    }

    const attemptTimeoutMs = Math.min(attempt.timeoutMs ?? remainingAfterDelayMs, remainingAfterDelayMs);
    const attemptController = new AbortController();
    let onParentAbort: (() => void) | null = null;
    if (options.parentSignal) {
      onParentAbort = () => attemptController.abort(options.parentSignal?.reason);
      options.parentSignal.addEventListener('abort', onParentAbort, { once: true });
    }

    let timer: ReturnType<typeof setTimeout> | null = null;
    const timeoutPromise = new Promise<never>((_, reject) => {
      timer = setTimeout(() => {
        attemptController.abort(new Error('Attempt TIMEOUT'));
        reject(Object.assign(new Error(`Model ${attempt.model} timed out after ${attemptTimeoutMs}ms (TIMEOUT)`), { status: 504 }));
      }, attemptTimeoutMs);
    });

    try {
      const result = await Promise.race([
        invokeFn(attempt, attemptController.signal),
        timeoutPromise,
      ]);
      return { result, modelUsed: attempt.model, attemptCount: i + 1 };
    } catch (err) {
      lastError = err;
      const transient = isTransientGeminiError(err);
      const info = parseGeminiErrorInfo(err);
      const timeLeft = totalBudgetMs - (Date.now() - startedAt);
      if (transient && !options.parentSignal?.aborted && timeLeft > 1500 && i < attempts.length - 1) {
        console.warn(
          `[analyze-trade-image retry]: ${attempt.model} (${info.status} ${info.statusText || 'TRANSIENT'}), sonraki denemeye geçiliyor...`
        );
        continue;
      }
      throw err;
    } finally {
      if (timer) clearTimeout(timer);
      if (options.parentSignal && onParentAbort) {
        options.parentSignal.removeEventListener('abort', onParentAbort);
      }
    }
  }
  throw lastError;
}

const VALID_TIMEFRAMES = new Set([
  'M1', 'M3', 'M5', 'M15', 'M30', 'H1', 'H2', 'H4', 'D1', 'W1', 'MN1'
]);

const VALID_SESSIONS = new Set(['London', 'New York', 'Asia', 'Overlap']);

export function normalizePair(raw: unknown): string | null {
  if (typeof raw !== 'string') return null;
  let cleaned = raw.trim().toUpperCase();
  if (!cleaned) return null;
  if (cleaned.includes(':')) {
    cleaned = cleaned.split(':').pop() || '';
  }
  cleaned = cleaned.replace(/\s+/g, '').replace(/\.P$/i, '');
  if (!cleaned || cleaned.length > 24) return null;
  return cleaned;
}

export function normalizeTimeframe(raw: unknown): string | null {
  if (typeof raw !== 'string') return null;
  const s = raw.trim().toUpperCase().replace(/\s+/g, '');
  if (!s) return null;
  if (VALID_TIMEFRAMES.has(s)) return s;

  const aliases: Record<string, string> = {
    '1': 'M1', '1M': 'M1', '1DK': 'M1', '1MIN': 'M1',
    '3': 'M3', '3M': 'M3', '3DK': 'M3', '3MIN': 'M3',
    '5': 'M5', '5M': 'M5', '5DK': 'M5', '5MIN': 'M5',
    '15': 'M15', '15M': 'M15', '15DK': 'M15', '15MIN': 'M15',
    '30': 'M30', '30M': 'M30', '30DK': 'M30', '30MIN': 'M30',
    '60': 'H1', '1H': 'H1', '1S': 'H1', '1SA': 'H1', '1SAAT': 'H1',
    '120': 'H2', '2H': 'H2', '2S': 'H2', '2SA': 'H2', '2SAAT': 'H2',
    '240': 'H4', '4H': 'H4', '4S': 'H4', '4SA': 'H4', '4SAAT': 'H4',
    'D': 'D1', '1D': 'D1', '1G': 'D1', 'DAILY': 'D1',
    'W': 'W1', '1W': 'W1', 'WEEKLY': 'W1',
    'M': 'MN1', 'MN': 'MN1', '1MO': 'MN1', 'MONTHLY': 'MN1'
  };
  return aliases[s] || null;
}

export function normalizeDirection(raw: unknown): 'Long' | 'Short' | null {
  if (typeof raw !== 'string') return null;
  const u = raw.trim().toLowerCase();
  if (u === 'long' || u === 'buy' || u === 'alış' || u === 'alis') return 'Long';
  if (u === 'short' || u === 'sell' || u === 'satış' || u === 'satis') return 'Short';
  return null;
}

export function normalizeResult(raw: unknown): 'Win' | 'Loss' | 'BE' | null {
  if (typeof raw !== 'string') return null;
  const u = raw.trim().toLowerCase();
  if (u === 'win' || u === 'kazan' || u === 'tp' || u === 'profit') return 'Win';
  if (u === 'loss' || u === 'kayıp' || u === 'kayip' || u === 'sl') return 'Loss';
  if (u === 'be' || u === 'breakeven' || u === 'başabaş' || u === 'basabas') return 'BE';
  return null;
}

export function normalizeSession(raw: unknown, hhmm: string | null): string | null {
  if (typeof raw === 'string') {
    const s = raw.trim().toLowerCase();
    if (s.includes('overlap')) return 'Overlap';
    if (s.includes('new york') || s === 'ny') return 'New York';
    if (s.includes('london') || s.includes('londra')) return 'London';
    if (s.includes('asia') || s.includes('asya') || s.includes('tokyo')) return 'Asia';
    if (VALID_SESSIONS.has(raw.trim())) return raw.trim();
  }
  if (hhmm) {
    const parts = hhmm.split(':').map(Number);
    const h = parts[0];
    if (Number.isFinite(h)) {
      if (h >= 0 && h < 8) return 'Asia';
      if (h >= 8 && h < 13) return 'London';
      if (h >= 13 && h < 17) return 'Overlap';
      return 'New York';
    }
  }
  return null;
}

export function normalizeDate(raw: unknown): string | null {
  if (typeof raw !== 'string') return null;
  const trimmed = raw.trim();
  const m = trimmed.match(/^(\d{4})-(\d{1,2})-(\d{1,2})$/);
  if (!m) return null;
  const y = Number(m[1]);
  const mo = Number(m[2]);
  const d = Number(m[3]);
  if (y < 2000 || y > 2100 || mo < 1 || mo > 12 || d < 1 || d > 31) return null;
  return `${String(y).padStart(4, '0')}-${String(mo).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
}

export function normalizeTime(raw: unknown): string | null {
  if (typeof raw !== 'string') return null;
  const trimmed = raw.trim();
  const m = trimmed.match(/(\d{1,2}):(\d{2})/);
  if (!m) return null;
  const h = Number(m[1]);
  const mi = Number(m[2]);
  if (h < 0 || h > 23 || mi < 0 || mi > 59) return null;
  return `${String(h).padStart(2, '0')}:${String(mi).padStart(2, '0')}`;
}

export function toFiniteNumber(raw: unknown): number | null {
  if (typeof raw === 'number') return Number.isFinite(raw) ? raw : null;
  if (typeof raw === 'string') {
    const cleaned = raw.replace(/[\s,$€₺+]/g, '').replace(',', '.');
    if (!cleaned) return null;
    const n = Number(cleaned);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

const VALID_UNCERTAIN_FIELDS = new Set([
  'pair',
  'direction',
  'datetime',
  'timeframe',
  'session',
  'entry',
  'exit',
  'sl',
  'risk',
  'pnl',
  'result',
]);

export function normalizeUncertainFields(raw: unknown): string[] {
  if (!Array.isArray(raw)) return [];
  const out = new Set<string>();
  for (const item of raw) {
    if (typeof item !== 'string') continue;
    const key = item.trim();
    if (!key) continue;
    const lower = key.toLowerCase();
    if (lower === 'date' || lower === 'time' || lower === 'datetime') {
      out.add('datetime');
    } else if (lower === 'riskpct' || lower === 'risk') {
      out.add('risk');
    } else if (VALID_UNCERTAIN_FIELDS.has(lower)) {
      out.add(lower);
    }
  }
  return Array.from(out);
}

export function sanitizeTradeNotes(raw: unknown): string {
  if (typeof raw !== 'string') return '';
  let text = raw.trim();
  if (!text) return '';

  const sentences = text
    .split(/(?<=[.!?])\s+|\n+/)
    .map(s => s.trim())
    .filter(Boolean);

  const kept: string[] = [];
  for (let s of sentences) {
    if (/\b(kullanıcı|kullanicı|kullanici|trader|yatırımcı|yatirimci)\b/i.test(s)) {
      if (/paritesinde|zaman diliminde|risk[- ]?kazanç|pozisyon\s+kur|pozisyon\s+aç/i.test(s)) {
        continue;
      }
      s = s
        .replace(/^(kullanıcı|trader|yatırımcı)(nın|nun|n)?[,:\s]+/i, '')
        .replace(/\b(kullanıcı|trader|yatırımcı)(ya|ye|yı|yi|nın|nun|dan|den|lar|ler)?\b/gi, '')
        .replace(/\s{2,}/g, ' ')
        .trim();
    }
    if (/paritesinde\b.*\bzaman diliminde\b/i.test(s)) continue;
    if (/risk[- ]?kazanç\s+oranına\s+sahip\s+(bir\s+)?(long|short)\s+pozisyon/i.test(s)) continue;
    if (!s || s.length < 4) continue;
    kept.push(s.charAt(0).toUpperCase() + s.slice(1));
  }

  return kept.join(' ').trim();
}

function buildSystemInstruction(): string {
  return [
    'Sen profesyonel bir finansal grafik analiz ve veri çıkarım motorusun. Görevin, yüklenen işlem (trade) grafik ekran görüntüsünden yalnızca kesin form verilerini ve kısa teknik yürütme notunu çıkarmaktır.',
    '',
    '1. TARİH VE SAAT OKUMA (ÇOK ÖNEMLİ):',
    '- Grafiğin EN ALTINDAKİ X-ekseni (zaman ekseni) üzerindeki saat ve tarih etiketlerini dikkatle incele.',
    '- Koyu/renkli crosshair saat rozetine (örn. "16:54", "11 Kas \'26 16:54", "2026-11-11 16:54"), pozisyon kutusunun sol kenarının (giriş mumu) dikey olarak denk geldiği alttaki X-ekseni saat işaretlerine ve sağ alt köşedeki grafik saatine bak.',
    '- Alttaki zaman ekseninde saat (HH:mm) okunabiliyorsa ASLA time alanını null bırakma. Tarih (gün/ay) görünmüyorsa bile saati mutlaka time alanına "HH:mm" (24 saat) olarak yaz.',
    '- Tarihte yıl yazmıyorsa içinde bulunulan yılı (2026) kabul et; gün ve ay yoksa date alanını null bırakıp time alanını yine de doldur.',
    '',
    '2. FİYAT VE YÖN TUTARLILIĞI:',
    '- TradingView/cTrader/MT5 Long/Short pozisyon kutularında yeşil alan TP (hedef), kırmızı alan SL (stop) ve orta çizgi Entry (giriş) fiyatıdır.',
    '- Yeşil bölge girişin ÜSTÜNDE ise direction="Long", yeşil bölge girişin ALTINDA ise direction="Short" olarak belirle.',
    '- Entry, Exit ve SL değerlerini yalnızca sağ fiyat eksenindeki etiketlerden veya pozisyon/emir çizgilerinden sayı olarak oku.',
    '- İşlem hedefte kapandıysa Exit=TP, stop olduysa Exit=SL olur. Long Win için exit > entry; Long Loss için exit < entry; Short Win için exit < entry; Short Loss için exit > entry kuralına uy.',
    '- TradingView Long/Short pozisyon kutusundaki hedef/stop mesafe yüzdelerini (örn. +0.17%, -0.09%) ASLA riskPct olarak alma. Yalnızca açıkça "Account Risk %" veya hesap risk yüzdesi yazıyorsa oku, aksi halde riskPct ve pnl için null döndür.',
    '',
    '3. İŞLEM NOTLARI (notes ALANI) — KESİN KURALLAR:',
    '- ASLA "Kullanıcı", "Trader", "Yatırımcı" ifadelerini veya "...kurmuş", "...açmış", "...hedeflemiş" gibi üçüncü tekil anlatımı kullanma.',
    '- ASLA parite adını, zaman dilimini (M1, M15 vb.), yönü (Long/Short) veya R:R oranını (1:2.32 vb.) tekrar eden arka plan özet cümlesi yazma. Örneğin: "Kullanıcı, NAS100 paritesinde M1 zaman diliminde 1:2.32 risk-kazanç oranına sahip bir Short pozisyon kurmuş." GİBİ BİR CÜMLE KESİNLİKLE YASAKTIR.',
    '- Yalnızca grafikte teknik olarak ne yapıldığını kısa, doğrudan ve edilgen işlem günlüğü diliyle 1-2 cümlede yaz (örnek: "Üst likidite alımı sonrası oluşan FVG bölgesinden tepkiyle işleme girildi. Son swing tepesi üzerinde stop konumlandırılıp alt likidite seviyesinde kâr alındı.").',
    '',
    '4. EMİN OLUNAMAYAN ALANLAR (uncertainFields ALANI):',
    '- Görselde silik, kesik, kısmen görünen, başka etiketlerle çakışan veya tam emin olamadan çıkardığın alanların adlarını uncertainFields dizisine ekle.',
    '- Kullanabileceğin alan adları: "pair", "direction", "date", "time", "timeframe", "session", "entry", "exit", "sl", "riskPct", "pnl", "result".',
    '- Ekranda net ve kesin biçimde okuduğun alanları uncertainFields içine EKLEME.'
  ].join('\n');
}

const TRADE_IMAGE_RESPONSE_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    pair: {
      type: Type.STRING,
      nullable: true,
      description: 'Büyük harfli parite veya enstrüman sembolü (örn. XAUUSD, NAS100, EURUSD, US30). Borsa öneki olmadan.'
    },
    direction: {
      type: Type.STRING,
      nullable: true,
      description: 'Yalnızca "Long" veya "Short". Okunamıyorsa null.'
    },
    date: {
      type: Type.STRING,
      nullable: true,
      description: 'Alt zaman ekseninden okunan işlem tarihi YYYY-MM-DD formatında. Tarih okunamıyorsa null.'
    },
    time: {
      type: Type.STRING,
      nullable: true,
      description: 'Alt zaman ekseninden, crosshair rozetinden, giriş hizasından veya sağ alt saatten okunan 24 saatlik saat (HH:mm). Altta saat varsa mutlaka doldur.'
    },
    timeframe: {
      type: Type.STRING,
      nullable: true,
      description: 'Yalnızca şunlardan biri: M1, M3, M5, M15, M30, H1, H2, H4, D1, W1, MN1. Okunamıyorsa null.'
    },
    session: {
      type: Type.STRING,
      nullable: true,
      description: 'Yalnızca şunlardan biri: Asia, London, Overlap, New York. Okunamıyorsa null.'
    },
    entry: {
      type: Type.NUMBER,
      nullable: true,
      description: 'Giriş fiyatı (sayısal). Okunamıyorsa null.'
    },
    exit: {
      type: Type.NUMBER,
      nullable: true,
      description: 'Çıkış / Kapanış fiyatı (sayısal). Okunamıyorsa null.'
    },
    sl: {
      type: Type.NUMBER,
      nullable: true,
      description: 'Stop Loss fiyatı (sayısal). Okunamıyorsa null.'
    },
    riskPct: {
      type: Type.NUMBER,
      nullable: true,
      description: 'Yalnızca açıkça hesap risk yüzdesi etiketi varsa döndür (TradingView hedef/stop fiyat mesafe yüzdelerini ASLA alma); yoksa null.'
    },
    pnl: {
      type: Type.NUMBER,
      nullable: true,
      description: 'Ekranda açıkça dolar kâr/zarar yazıyorsa sayısal değeri (kayıpsa negatif), yazmıyorsa null.'
    },
    result: {
      type: Type.STRING,
      nullable: true,
      description: 'Yalnızca "Win", "Loss" veya "BE". Okunamıyorsa null.'
    },
    notes: {
      type: Type.STRING,
      nullable: true,
      description: '"Kullanıcı" kelimesi ve parite/timeframe/R:R tekrarı OLMADAN yalnızca grafikte teknik olarak ne yapıldığını anlatan kısa 1-2 cümlelik not.'
    },
    uncertainFields: {
      type: Type.ARRAY,
      nullable: true,
      items: { type: Type.STRING },
      description: 'Görselde silik, kesik veya belirsiz olduğu için kullanıcının kontrol etmesi gereken alan adları listesi (örn. pair, direction, date, time, timeframe, session, entry, exit, sl, riskPct, pnl, result).'
    }
  }
};

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '14mb' }));

  app.post('/api/analyze-trade-image', async (req: Request, res: Response) => {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      return res.status(500).json({
        error: 'Sunucuda GEMINI_API_KEY tanımlı değil. Ayarlar > Secrets panelinden anahtarı kontrol edin.'
      });
    }

    const { imageBase64, mimeType } = req.body || {};
    if (typeof imageBase64 !== 'string' || imageBase64.trim().length < 32) {
      return res.status(400).json({ error: 'Geçersiz görsel verisi gönderildi.' });
    }

    const safeMime =
      typeof mimeType === 'string' && mimeType.startsWith('image/')
        ? mimeType
        : 'image/png';
    const cleanBase64 = imageBase64.replace(/^data:image\/[a-zA-Z0-9+.-]+;base64,/, '');

    const requestController = new AbortController();
    const onResClose = () => {
      if (!res.writableEnded) {
        requestController.abort(new Error('Client closed connection'));
      }
    };
    res.on('close', onResClose);

    try {
      const ai = new GoogleGenAI({
        apiKey,
        httpOptions: {
          timeout: ANALYZE_SERVER_BUDGET_MS,
          headers: {
            'User-Agent': 'aistudio-build',
          },
        },
      });

      const { result: response } = await runWithModelFallback(
        async (attempt, signal) => {
          const config: {
            systemInstruction: string;
            thinkingConfig?: { thinkingLevel: ThinkingLevel };
            responseMimeType: string;
            responseSchema: typeof TRADE_IMAGE_RESPONSE_SCHEMA;
            abortSignal?: AbortSignal;
          } = {
            systemInstruction: buildSystemInstruction(),
            responseMimeType: 'application/json',
            responseSchema: TRADE_IMAGE_RESPONSE_SCHEMA,
            abortSignal: signal,
          };
          if (attempt.thinkingLevel) {
            config.thinkingConfig = { thinkingLevel: attempt.thinkingLevel };
          }
          const gen = await ai.models.generateContent({
            model: attempt.model,
            contents: {
              parts: [
                {
                  inlineData: {
                    mimeType: safeMime,
                    data: cleanBase64,
                  },
                },
                {
                  text:
                    'Bu grafik ekran görüntüsündeki tüm işlem bilgilerini çıkar. Alt X-eksenindeki saat etiketlerine, giriş/çıkış/SL fiyatlarına dikkatle bak ve notes alanında yalnızca ne yapıldığını teknik olarak yaz (kesinlikle "Kullanıcı" kelimesini ve parite/timeframe/R:R özet cümlesini yazma).',
                },
              ],
            },
            config,
          });
          if (!gen.text || !gen.text.trim()) {
            throw new Error('empty response');
          }
          return gen;
        },
        MODEL_ATTEMPTS,
        defaultSleep,
        { totalBudgetMs: ANALYZE_SERVER_BUDGET_MS, parentSignal: requestController.signal }
      );

      const rawText = (response.text ?? '{}').trim();
      let parsed: Record<string, unknown> = {};
      try {
        parsed = JSON.parse(rawText);
      } catch {
        const jsonMatch = rawText.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          parsed = JSON.parse(jsonMatch[0]);
        }
      }

      const pair = normalizePair(parsed.pair);
      let direction = normalizeDirection(parsed.direction);
      const date = normalizeDate(parsed.date);
      const time = normalizeTime(parsed.time);
      const datetime = date && time ? `${date}T${time}` : null;
      const timeframe = normalizeTimeframe(parsed.timeframe);
      const session = normalizeSession(parsed.session, time);
      const entry = toFiniteNumber(parsed.entry);
      const exit = toFiniteNumber(parsed.exit);
      const sl = toFiniteNumber(parsed.sl);
      const riskPct = toFiniteNumber(parsed.riskPct);
      const pnl = toFiniteNumber(parsed.pnl);
      let result = normalizeResult(parsed.result);

      if (!direction && entry !== null && sl !== null && entry !== sl) {
        direction = sl < entry ? 'Long' : 'Short';
      }

      if (!result && entry !== null && exit !== null && direction) {
        if (Math.abs(exit - entry) === 0) {
          result = 'BE';
        } else if (direction === 'Long') {
          result = exit > entry ? 'Win' : 'Loss';
        } else {
          result = exit < entry ? 'Win' : 'Loss';
        }
      }

      const notes = sanitizeTradeNotes(parsed.notes);
      const uncertainFields = normalizeUncertainFields(parsed.uncertainFields);

      return res.json({
        extracted: {
          pair,
          direction,
          date,
          time,
          datetime,
          timeframe,
          session,
          entry,
          exit,
          sl,
          riskPct: riskPct !== null && riskPct > 0 ? riskPct : null,
          pnl,
          result,
          notes,
          uncertainFields,
        },
      });
    } catch (err: unknown) {
      if (requestController.signal.aborted || res.headersSent) {
        return;
      }
      const info = parseGeminiErrorInfo(err);
      const friendly = formatFriendlyGeminiError(err);
      console.error(`[analyze-trade-image error] (${info.status} ${info.statusText}):`, info.message);
      return res.status(friendly.status).json({
        error: friendly.error,
        retryable: isTransientGeminiError(err),
      });
    } finally {
      res.off('close', onResClose);
    }
  });

  const isProd = process.env.NODE_ENV === 'production';
  const distDir = path.resolve(__dirname, 'dist');

  if (isProd && fs.existsSync(distDir)) {
    app.use(express.static(distDir));
    app.get('/{*splat}', (_req: Request, res: Response) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  } else {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Prop Journal server running on http://0.0.0.0:${PORT}`);
  });
}

const isDirectRun = Boolean(
  process.argv[1] && path.resolve(process.argv[1]) === path.resolve(__filename)
);

if (isDirectRun) {
  startServer().catch((err) => {
    console.error('Server failed to start:', err);
    process.exit(1);
  });
}
