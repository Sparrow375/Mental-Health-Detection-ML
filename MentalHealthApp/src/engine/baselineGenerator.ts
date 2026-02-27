import { FEATURE_KEYS, FEATURE_DEFAULTS, PersonalityVector } from './types';

function randomNormal(mean: number, sd: number): number {
    const u1 = Math.random(), u2 = Math.random();
    return Math.max(0, mean + Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * sd);
}

export function generateSyntheticBaseline(): PersonalityVector {
    const days = 28; const NOISE = 0.12;
    const accum: Record<string, number[]> = {};
    for (const k of FEATURE_KEYS) accum[k] = [];
    for (let d = 0; d < days; d++)
        for (const k of FEATURE_KEYS) {
            const m = (FEATURE_DEFAULTS as any)[k] as number;
            accum[k].push(randomNormal(m, m * NOISE));
        }
    const means: Record<string, number> = {};
    const variances: Record<string, number> = {};
    for (const k of FEATURE_KEYS) {
        const vals = accum[k];
        const m = vals.reduce((a, b) => a + b, 0) / vals.length;
        const s = Math.sqrt(vals.reduce((s, v) => s + (v - m) ** 2, 0) / vals.length);
        means[k] = m; variances[k] = Math.max(s, 0.001);
    }
    return { ...(means as any), variances } as PersonalityVector;
}

export function buildBaselineFromInputs(inputs: Partial<Record<string, number>>): PersonalityVector {
    const result: Record<string, number> = {};
    const variances: Record<string, number> = {};
    for (const k of FEATURE_KEYS) {
        const val = inputs[k] ?? (FEATURE_DEFAULTS as any)[k];
        result[k] = val; variances[k] = Math.max(val * 0.12, 0.001);
    }
    return { ...(result as any), variances } as PersonalityVector;
}
