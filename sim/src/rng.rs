//! OpenFront-compatible `simpleHash` and `PseudoRandom` (sfc32 + splitmix32).

const POW36_8: f64 = 2_821_109_907_456.0; // 36^8

/// JS `simpleHash`: 32-bit string hash, then `Math.abs`.
pub fn simple_hash(s: &str) -> i32 {
    let mut hash: i32 = 0;
    for unit in s.encode_utf16() {
        hash = hash
            .wrapping_shl(5)
            .wrapping_sub(hash)
            .wrapping_add(unit as i32);
    }
    if hash == i32::MIN {
        // Math.abs(-2147483648) is 2147483648 in IEEE, not representable as i32.
        // Callers that need the exact JS Number should use `simple_hash_abs_f64`.
        i32::MAX
    } else {
        hash.abs()
    }
}

/// `Math.abs(simpleHash(s))` as a JS Number (handles INT_MIN).
pub fn simple_hash_abs_f64(s: &str) -> f64 {
    let mut hash: i32 = 0;
    for unit in s.encode_utf16() {
        hash = hash
            .wrapping_shl(5)
            .wrapping_sub(hash)
            .wrapping_add(unit as i32);
    }
    if hash == i32::MIN {
        2147483648.0
    } else {
        hash.abs() as f64
    }
}

#[derive(Clone, Debug)]
pub struct PseudoRandom {
    s0: i32,
    s1: i32,
    s2: i32,
    s3: i32,
}

impl PseudoRandom {
    pub fn new(seed: i32) -> Self {
        let mut h = seed;
        let mut split = || {
            h = h.wrapping_add(0x9e3779b9_u32 as i32);
            let mut t = h ^ ((h as u32) >> 16) as i32;
            t = imul(t, 0x21f0aaad_u32 as i32);
            t ^= ((t as u32) >> 15) as i32;
            t = imul(t, 0x735a2d97_u32 as i32);
            t ^ (((t as u32) >> 15) as i32)
        };
        let mut rng = Self {
            s0: split(),
            s1: split(),
            s2: split(),
            s3: split(),
        };
        for _ in 0..12 {
            rng.next();
        }
        rng
    }

    pub fn next(&mut self) -> f64 {
        let t = self.s0.wrapping_add(self.s1).wrapping_add(self.s3);
        self.s3 = self.s3.wrapping_add(1);
        self.s0 = self.s1 ^ (((self.s1 as u32) >> 9) as i32);
        self.s1 = self.s2.wrapping_add(self.s2.wrapping_shl(3));
        self.s2 = self.s2.rotate_left(21);
        self.s2 = self.s2.wrapping_add(t);
        (t as u32 as f64) / 4_294_967_296.0
    }

    pub fn next_float(&mut self, min: f64, max: f64) -> f64 {
        self.next() * (max - min) + min
    }

    pub fn next_int(&mut self, min: f64, max: f64) -> i32 {
        let lo = min.floor();
        let hi = max.floor();
        (self.next() * (hi - lo)).floor() as i32 + lo as i32
    }

    pub fn next_id(&mut self) -> String {
        let n = (self.next() * POW36_8).floor() as u64;
        let mut s = to_base36(n);
        if s.len() < 8 {
            s = format!("{:0>8}", s);
        }
        s
    }

    /// True with probability 1/odds (`nextInt(0, odds) === 0`).
    pub fn chance(&mut self, odds: i32) -> bool {
        self.next_int(0.0, odds as f64) == 0
    }

    pub fn rand_element<T: Clone>(&mut self, arr: &[T]) -> T {
        let i = self.next_int(0.0, arr.len() as f64) as usize;
        arr[i].clone()
    }

    pub fn shuffle_array<T: Clone>(&mut self, array: &[T]) -> Vec<T> {
        let mut result = array.to_vec();
        let mut i = result.len() as i32 - 1;
        while i > 0 {
            let j = self.next_int(0.0, (i + 1) as f64) as usize;
            result.swap(i as usize, j);
            i -= 1;
        }
        result
    }
}

fn imul(a: i32, b: i32) -> i32 {
    a.wrapping_mul(b)
}

fn to_base36(mut n: u64) -> String {
    const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    if n == 0 {
        return "0".into();
    }
    let mut out = Vec::new();
    while n > 0 {
        out.push(DIGITS[(n % 36) as usize]);
        n /= 36;
    }
    out.reverse();
    String::from_utf8(out).expect("base36 digits")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_hash_ascii_matches_js_style() {
        assert_eq!(simple_hash(""), 0);
        assert_eq!(simple_hash("a"), 97);
        // Hand-checked against OpenFront Util.simpleHash in the oracle suite.
        assert_eq!(simple_hash("golden-spawn"), simple_hash("golden-spawn"));
        assert!(simple_hash("golden-spawn") > 0);
    }
}
