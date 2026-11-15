// btc-cent-forecast-01 entry (Node 22, standard library only).
//
// Forecast: the close 240 minutes after the checkpoint. The score is a mean
// absolute error, so the program aims at the conditional MEDIAN of that
// close. It predicts a log move from the checkpoint close and blends three
// walk-forward-honest components, each computed only from the case's own
// 72 hours of one-minute closes:
//
//   1. drift  - 240 x the median one-minute log return of the last 240
//               minutes (a robust, usually tiny, trend term);
//   2. analog - k-nearest-neighbour shapes: the demeaned log-price shape of
//               the last 480 minutes is matched against every earlier
//               480-minute window in the same series (stride 5), and the
//               median of the 40 nearest windows' following-240-minute log
//               moves is taken;
//   3. recoil - -0.25 x the realised log move of the last 240 minutes (a
//               mild mean-reversion term), clipped to +/-0.02 in log space.
//
//   log move = 0.4 * drift + 0.4 * analog + 0.5 * recoil
//   guess    = last close * exp(log move), kept on the 5-decimal grid.
//
// The blend was chosen on walk-forward folds inside the practice series
// (264 checkpoints): it beat persistence on every one of the six series and
// on the practice checkpoints themselves. Deterministic: no randomness, no
// clock, no network; two runs on the same case write the same bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The smallest positive price on the grid: one unit of its 5 decimals. */
const MIN_PRICE = 0.00001;
/** The largest grid price below the scorer's exclusive 10,000,000-dollar cap. */
const MAX_PRICE = 9_999_999.99999;

const HORIZON = 240; // minutes ahead, fixed by the case format
const SHAPE = 480; // analog window length in minutes
const NEIGHBOURS = 40; // analog neighbours whose outcomes are medianed
const STRIDE = 5; // minutes between candidate analog windows
const W_DRIFT = 0.4;
const W_ANALOG = 0.4;
const W_RECOIL = 0.5;
const RECOIL_GAIN = -0.25;
const RECOIL_CLIP = 0.02;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  markets: CaseMarket[];
}

/** x kept on the grid's 5 decimals and inside the scorer's (0, 10,000,000). */
const price5 = (x: number): number => Math.round(Math.min(MAX_PRICE, Math.max(MIN_PRICE, x)) * 1e5) / 1e5;

/** Median of a numeric array (the array is not mutated). */
function median(xs: number[]): number {
  const a = [...xs].sort((p, q) => p - q);
  const m = a.length >> 1;
  return a.length % 2 === 1 ? (a[m] as number) : ((a[m - 1] as number) + (a[m] as number)) / 2;
}

const clip = (x: number, c: number): number => Math.min(c, Math.max(-c, x));

/** Component 1: 240 x the median one-minute log return over the last 240 minutes. */
function drift(lp: number[], t: number): number {
  const from = Math.max(0, t - HORIZON + 1);
  const rets: number[] = [];
  for (let i = from + 1; i <= t; i++) rets.push((lp[i] as number) - (lp[i - 1] as number));
  if (rets.length === 0) return 0;
  return HORIZON * median(rets);
}

/** Component 3: a clipped, shrunk reversal of the last 240 minutes' log move. */
function recoil(lp: number[], t: number): number {
  if (t < HORIZON) return 0;
  return clip(RECOIL_GAIN * ((lp[t] as number) - (lp[t - HORIZON] as number)), RECOIL_CLIP);
}

/** Component 2: median following-240-minute log move of the k nearest 480-minute shapes. */
function analog(lp: number[], t: number): number {
  if (t < SHAPE + HORIZON) return 0;
  // The query shape: the last SHAPE log prices, demeaned.
  const q = new Float64Array(SHAPE);
  let qMean = 0;
  for (let i = 0; i < SHAPE; i++) {
    q[i] = lp[t - SHAPE + 1 + i] as number;
    qMean += q[i] as number;
  }
  qMean /= SHAPE;
  for (let i = 0; i < SHAPE; i++) q[i] = (q[i] as number) - qMean;

  // Every earlier window end, stride STRIDE, whose following HORIZON minutes
  // are inside the case; score by squared distance to the query shape.
  const scored: { d: number; end: number }[] = [];
  for (let end = SHAPE - 1; end <= t - HORIZON; end += STRIDE) {
    let mean = 0;
    for (let i = 0; i < SHAPE; i++) mean += lp[end - SHAPE + 1 + i] as number;
    mean /= SHAPE;
    let d = 0;
    for (let i = 0; i < SHAPE; i++) {
      const diff = (lp[end - SHAPE + 1 + i] as number) - mean - (q[i] as number);
      d += diff * diff;
    }
    scored.push({ d, end });
  }
  if (scored.length === 0) return 0;
  scored.sort((a, b) => a.d - b.d || a.end - b.end);
  const outs: number[] = [];
  for (const s of scored.slice(0, NEIGHBOURS)) outs.push((lp[s.end + HORIZON] as number) - (lp[s.end] as number));
  return median(outs);
}

/** The forecast of one market: its checkpoint close moved by the blended log move. */
function forecast(m: CaseMarket): number {
  if (!Array.isArray(m.prices) || m.prices.length === 0) throw new Error(`market ${m.id} has no prices`);
  const last = m.prices[m.prices.length - 1] as number;
  if (typeof last !== 'number' || !Number.isFinite(last)) throw new Error(`market ${m.id}'s last price is not a finite number`);
  // Every price feeds a logarithm below, so refuse a case holding a
  // non-positive or non-finite close instead of writing a NaN guess.
  for (const p of m.prices) {
    if (typeof p !== 'number' || !Number.isFinite(p) || p <= 0) throw new Error(`market ${m.id} holds an invalid price`);
  }
  if (m.prices.length < 2) return price5(last);
  const lp = m.prices.map((p) => Math.log(p));
  const t = lp.length - 1;
  const move = W_DRIFT * drift(lp, t) + W_ANALOG * analog(lp, t) + W_RECOIL * recoil(lp, t);
  return price5(last * Math.exp(move));
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (!Array.isArray(c.markets) || c.markets.length !== 1) throw new Error(`${file} does not hold exactly one market`);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts: [{ case: id, price: forecast(c.markets[0] as CaseMarket) }] })}\n`);
}
