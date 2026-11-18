import test from 'node:test';
import assert from 'node:assert/strict';
import {
  normalizePair,
  normalizeTimeframe,
  normalizeDirection,
  normalizeResult,
  normalizeSession,
  normalizeDate,
  normalizeTime,
  toFiniteNumber,
  sanitizeTradeNotes,
  parseGeminiErrorInfo,
  isTransientGeminiError,
  formatFriendlyGeminiError,
  runWithModelFallback,
  MODEL_ATTEMPTS,
} from './server.ts';

test('normalizePair strips exchange prefixes and futures suffix', () => {
  assert.equal(normalizePair('NASDAQ:NAS100'), 'NAS100');
  assert.equal(normalizePair('OANDA:XAUUSD.P'), 'XAUUSD');
  assert.equal(normalizePair(' eurusd '), 'EURUSD');
  assert.equal(normalizePair(''), null);
});

test('normalizeTimeframe maps TradingView intervals and aliases', () => {
  assert.equal(normalizeTimeframe('1m'), 'M1');
  assert.equal(normalizeTimeframe('1'), 'M1');
  assert.equal(normalizeTimeframe('15dk'), 'M15');
  assert.equal(normalizeTimeframe('60'), 'H1');
  assert.equal(normalizeTimeframe('4h'), 'H4');
  assert.equal(normalizeTimeframe('M1'), 'M1');
});

test('normalizeTime and normalizeDate extract bottom axis values cleanly', () => {
  assert.equal(normalizeTime('16:54'), '16:54');
  assert.equal(normalizeTime('9:05 UTC'), '09:05');
  assert.equal(normalizeTime('25:99'), null);
  assert.equal(normalizeDate('2026-11-11'), '2026-11-11');
  assert.equal(normalizeDate('2026-9-4'), '2026-09-04');
  assert.equal(normalizeDate('11-11'), null);
});

test('normalizeSession infers session from clock when explicit session is absent', () => {
  assert.equal(normalizeSession(null, '04:30'), 'Asia');
  assert.equal(normalizeSession(null, '10:15'), 'London');
  assert.equal(normalizeSession(null, '14:45'), 'Overlap');
  assert.equal(normalizeSession(null, '18:20'), 'New York');
  assert.equal(normalizeSession('London Open', '18:20'), 'London');
});

test('normalizeDirection, normalizeResult, and toFiniteNumber handle standard values', () => {
  assert.equal(normalizeDirection('sell'), 'Short');
  assert.equal(normalizeDirection('Long'), 'Long');
  assert.equal(normalizeResult('tp'), 'Win');
  assert.equal(normalizeResult('Loss'), 'Loss');
  assert.equal(toFiniteNumber('$+1,240.50'), 1240.5);
  assert.equal(toFiniteNumber('$450.25'), 450.25);
  assert.equal(toFiniteNumber('-150'), -150);
});

test('sanitizeTradeNotes removes "kullanıcı" mentions and background metadata recap', () => {
  const input =
    'Kullanıcı, NAS100 paritesinde M1 zaman diliminde 1:2.32 risk-kazanç oranına sahip bir Short pozisyon kurmuş. Likidite alımı sonrası FVG bölgesinden tepkiyle işleme girildi.';
  const cleaned = sanitizeTradeNotes(input);
  assert.equal(/kullanıcı/i.test(cleaned), false);
  assert.equal(/paritesinde/i.test(cleaned), false);
  assert.equal(/risk[- ]?kazanç/i.test(cleaned), false);
  assert.equal(cleaned, 'Likidite alımı sonrası FVG bölgesinden tepkiyle işleme girildi.');
});

test('sanitizeTradeNotes keeps clean technical notes intact', () => {
  const clean = 'Üst likidite süpürülüp market yapısı aşağı kırıldıktan sonra FVG içinde Short giriş yapıldı. Son swing tepesi üzeri stop belirlendi.';
  assert.equal(sanitizeTradeNotes(clean), clean);
});

test('parseGeminiErrorInfo and formatFriendlyGeminiError extract nested 503 JSON cleanly', () => {
  const raw503 = new Error(
    '{"error":{"code":503,"message":"This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.","status":"UNAVAILABLE"}}'
  );
  const parsed = parseGeminiErrorInfo(raw503);
  assert.equal(parsed.status, 503);
  assert.equal(parsed.statusText, 'UNAVAILABLE');
  assert.equal(isTransientGeminiError(raw503), true);

  const friendly = formatFriendlyGeminiError(raw503);
  assert.equal(friendly.status, 503);
  assert.equal(friendly.error.includes('Tekrar Dene'), true);
  assert.equal(friendly.error.startsWith('{'), false);
});

test('runWithModelFallback retries transient 503 errors and falls back to lite model', async () => {
  const calledModels: string[] = [];
  const noopSleep = async () => {};
  const err503 = new Error(
    '{"error":{"code":503,"message":"This model is currently experiencing high demand.","status":"UNAVAILABLE"}}'
  );

  const out = await runWithModelFallback(
    async (attempt) => {
      calledModels.push(attempt.model);
      if (calledModels.length < 3) {
        throw err503;
      }
      return { text: '{"pair":"NAS100"}' };
    },
    MODEL_ATTEMPTS,
    noopSleep
  );

  assert.deepEqual(calledModels, [
    'gemini-4-hydrogen',
    'gemini-4-hydrogen',
    'gemini-3.1-flash-lite',
  ]);
  assert.equal(out.modelUsed, 'gemini-3.1-flash-lite');
  assert.equal(out.attemptCount, 3);
});

test('runWithModelFallback stops immediately on non-transient 403 error', async () => {
  let calls = 0;
  const noopSleep = async () => {};
  const err403 = Object.assign(new Error('PERMISSION_DENIED: invalid key'), { status: 403 });

  await assert.rejects(
    () =>
      runWithModelFallback(
        async () => {
          calls++;
          throw err403;
        },
        MODEL_ATTEMPTS,
        noopSleep
      ),
    /PERMISSION_DENIED/
  );
  assert.equal(calls, 1);
});

test('runWithModelFallback throws last transient error once all attempts exhaust', async () => {
  let calls = 0;
  const noopSleep = async () => {};
  const err503 = new Error('{"error":{"code":503,"message":"high demand","status":"UNAVAILABLE"}}');

  await assert.rejects(
    () =>
      runWithModelFallback(
        async () => {
          calls++;
          throw err503;
        },
        MODEL_ATTEMPTS,
        noopSleep
      ),
    /high demand/
  );
  assert.equal(calls, MODEL_ATTEMPTS.length);
});

