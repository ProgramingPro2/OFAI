//! V8/fdlibm `Math.pow` so hashed troop sends match official JS.

fn words(x: f64) -> (i32, u32) {
    let b = x.to_bits();
    ((b >> 32) as i32, b as u32)
}

fn set_low(x: f64, low: u32) -> f64 {
    f64::from_bits((x.to_bits() & 0xffff_ffff_0000_0000) | u64::from(low))
}

fn set_high(x: f64, high: i32) -> f64 {
    f64::from_bits(((high as u32 as u64) << 32) | (x.to_bits() & 0xffff_ffff))
}

fn high(x: f64) -> i32 {
    words(x).0
}

fn load_u32s(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn apply_ulp_corr(n: u32, bits: u64, minus: &[u32], plus: &[u32]) -> u64 {
    if minus.binary_search(&n).is_ok() {
        bits.wrapping_sub(1)
    } else if plus.binary_search(&n).is_ok() {
        bits.wrapping_add(1)
    } else {
        bits
    }
}

/// Node 22 / V8 12.4 `Math.pow` for OpenFront troop formulas (`x > 0`).
///
/// The fdlibm port is 1 ULP off for some integer bases. Official `maxTroops`
/// and `troopIncreaseRate` only ever raise an integer tile/troop count to
/// 0.6 or 0.73, so those cases are corrected to this host's Node bits.
pub fn js_pow(x: f64, y: f64) -> f64 {
    let raw = js_pow_fdlibm(x, y);
    if x < 1.0 || x != x.trunc() || x > u32::MAX as f64 {
        return raw;
    }
    let n = x as u32;
    let bits = raw.to_bits();
    let bits = if y == 0.6 {
        static M1: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
        static P1: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
        apply_ulp_corr(
            n,
            bits,
            M1.get_or_init(|| load_u32s(include_bytes!("js_pow_data/pow06_m1.bin"))),
            P1.get_or_init(|| load_u32s(include_bytes!("js_pow_data/pow06_p1.bin"))),
        )
    } else if y == 0.73 {
        static M1: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
        static P1: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
        apply_ulp_corr(
            n,
            bits,
            M1.get_or_init(|| load_u32s(include_bytes!("js_pow_data/pow73_m1.bin"))),
            P1.get_or_init(|| load_u32s(include_bytes!("js_pow_data/pow73_p1.bin"))),
        )
    } else {
        bits
    };
    f64::from_bits(bits)
}

/// fdlibm `__ieee754_pow` used as the base for integer ULP corrections.
fn js_pow_fdlibm(x: f64, y: f64) -> f64 {
    const BP: [f64; 2] = [1.0, 1.5];
    const DP_H: [f64; 2] = [0.0, 5.849_624_872_207_641_601_56e-1];
    const DP_L: [f64; 2] = [0.0, 1.350_039_202_129_748_971_28e-8];
    const L1: f64 = 5.999_999_999_999_946_487_25e-1;
    const L2: f64 = 4.285_714_285_785_501_842_52e-1;
    const L3: f64 = 3.333_333_298_183_774_329_18e-1;
    const L4: f64 = 2.727_281_238_085_340_064_89e-1;
    const L5: f64 = 2.306_607_457_755_617_540_67e-1;
    const L6: f64 = 2.069_750_178_003_384_177_84e-1;
    const P1: f64 = 1.666_666_666_666_660_190_37e-1;
    const P2: f64 = -2.777_777_777_701_559_338_42e-3;
    const P3: f64 = 6.613_756_321_437_934_361_17e-5;
    const P4: f64 = -1.653_390_220_546_525_153_90e-6;
    const P5: f64 = 4.138_136_797_057_238_460_39e-8;
    const LG2: f64 = 6.931_471_805_599_452_862_27e-1;
    const LG2_H: f64 = 6.931_471_824_645_996_093_75e-1;
    const LG2_L: f64 = -1.904_654_299_957_768_045_25e-9;
    const OVT: f64 = 8.008_566_259_537_294_437_2e-17;
    const CP: f64 = 9.617_966_939_259_755_543_29e-1;
    const CP_H: f64 = 9.617_967_009_544_372_558_59e-1;
    const CP_L: f64 = -7.028_461_650_952_758_265_16e-9;
    const TWO53: f64 = 9_007_199_254_740_992.0;

    if y == 0.0 {
        return 1.0;
    }
    if x == 1.0 {
        return 1.0;
    }
    if !x.is_finite() || !y.is_finite() || x <= 0.0 {
        return x.powf(y);
    }
    if y == 1.0 {
        return x;
    }
    if y == 2.0 {
        return x * x;
    }
    if y == 0.5 {
        return x.sqrt();
    }

    let (hx, _lx) = words(x);
    let ix = hx & 0x7fff_ffff;
    let ax = x.abs();

    let (t1, t2) = {
        let mut n = 0i32;
        let mut ax = ax;
        let mut ix = ix;
        if (ix as u32) < 0x0010_0000 {
            ax *= TWO53;
            n -= 53;
            ix = high(ax);
        }
        n += (ix >> 20) - 0x3ff;
        let j = ix & 0x000f_ffff;
        let (k, mut ix) = if j <= 0x3988e {
            (0, j | 0x3ff0_0000)
        } else if j < 0xbb67a {
            (1, j | 0x3ff0_0000)
        } else {
            n += 1;
            (0, (j | 0x3ff0_0000) - 0x0010_0000)
        };
        ax = set_high(ax, ix);
        let u = ax - BP[k];
        let v = 1.0 / (ax + BP[k]);
        let ss = u * v;
        let s_h = set_low(ss, 0);
        let mut t_h = 0.0;
        t_h = set_high(
            t_h,
            ((ix >> 1) | 0x2000_0000) + 0x0008_0000 + ((k as i32) << 18),
        );
        let t_l = ax - (t_h - BP[k]);
        let s_l = v * ((u - s_h * t_h) - s_h * t_l);
        let s2 = ss * ss;
        let mut r = s2 * s2 * (L1 + s2 * (L2 + s2 * (L3 + s2 * (L4 + s2 * (L5 + s2 * L6)))));
        r += s_l * (s_h + ss);
        let s2 = s_h * s_h;
        let t_h = set_low(3.0 + s2 + r, 0);
        let t_l = r - ((t_h - 3.0) - s2);
        let u = s_h * t_h;
        let v = s_l * t_h + t_l * ss;
        let p_h = set_low(u + v, 0);
        let p_l = v - (p_h - u);
        let z_h = CP_H * p_h;
        let z_l = CP_L * p_h + p_l * CP + DP_L[k];
        let t = n as f64;
        let t1 = set_low(((z_h + z_l) + DP_H[k]) + t, 0);
        let t2 = z_l - (((t1 - t) - DP_H[k]) - z_h);
        (t1, t2)
    };

    let y1 = set_low(y, 0);
    let p_l = (y - y1) * t1 + y * t2;
    let p_h = y1 * t1;
    let z = p_l + p_h;
    let (j, i) = words(z);
    if j >= 0x4090_0000 {
        if ((j - 0x4090_0000) | i as i32) != 0 {
            return f64::INFINITY;
        }
        if p_l + OVT > z - p_h {
            return f64::INFINITY;
        }
    } else if (j & 0x7fff_ffff) >= 0x4090_cc00 {
        if ((j - 0xc090_cc00_u32 as i32) | i as i32) != 0 {
            return 0.0;
        }
        if p_l <= z - p_h {
            return 0.0;
        }
    }

    let i = j & 0x7fff_ffff;
    let mut k = (i >> 20) - 0x3ff;
    let mut n = 0i32;
    let mut p_h = p_h;
    if i > 0x3fe0_0000 {
        n = j + (0x0010_0000 >> (k + 1));
        k = ((n & 0x7fff_ffff) >> 20) - 0x3ff;
        let t = set_high(0.0, n & !(0x000f_ffff >> k));
        n = ((n & 0x000f_ffff) | 0x0010_0000) >> (20 - k);
        if j < 0 {
            n = -n;
        }
        p_h -= t;
    }
    let t = set_low(p_l + p_h, 0);
    let u = t * LG2_H;
    let v = (p_l - (t - p_h)) * LG2 + t * LG2_L;
    let z = u + v;
    let w = v - (z - u);
    let t = z * z;
    let t1 = z - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    let r = (z * t1) / (t1 - 2.0) - (w + z * w);
    let mut z = 1.0 - (r - z);
    let mut jj = high(z);
    jj += n << 20;
    if (jj >> 20) <= 0 {
        z = libm_scalbn(z, n);
    } else {
        z = set_high(z, jj);
    }
    z
}

fn libm_scalbn(x: f64, n: i32) -> f64 {
    x * 2f64.powi(n)
}

#[cfg(test)]
mod tests {
    use super::js_pow;

    #[test]
    fn matches_v8_pow_for_diplomacy_send() {
        // Node Math.pow(702, 0.6) bits collected on this host.
        let p = js_pow(702.0, 0.6);
        assert_eq!(p.to_bits(), 0x4049_837d_53b9_dd46);
    }

    #[test]
    fn matches_node_pow_for_warships_combat_send() {
        // Node Math.pow(6070, 0.6) — p4 land send at tick 260.
        assert_eq!(js_pow(6070.0, 0.6).to_bits(), 0x4067_457c_fc9c_5aff);
    }
}
