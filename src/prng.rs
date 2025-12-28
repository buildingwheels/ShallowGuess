// Pseudorandom number generator based on ChaCha20 by Daniel J. Bernstein

pub struct RandGenerator {
    state: [u32; 16],
    keystream: [u8; 64],
    position: usize,
}

impl RandGenerator {
    pub fn new(seed: u64) -> Self {
        let mut key = [0u8; 32];
        let mut nonce = [0u8; 12];

        let mut s = seed;
        for i in 0..4 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bytes = s.to_le_bytes();
            key[i * 8..i * 8 + 8].copy_from_slice(&bytes);
        }

        nonce[0..8].copy_from_slice(&seed.to_le_bytes());
        nonce[8..12].copy_from_slice(&(seed.wrapping_mul(0x9e3779b97f4a7c15)).to_le_bytes()[0..4]);

        Self::new_with_key_and_nonce(&key, &nonce)
    }

    pub fn new_with_key_and_nonce(key: &[u8; 32], nonce: &[u8; 12]) -> Self {
        let mut state = [0u32; 16];

        state[0] = 0x61707865;
        state[1] = 0x3320646e;
        state[2] = 0x79622d32;
        state[3] = 0x6b206574;

        for i in 0..8 {
            state[4 + i] =
                u32::from_le_bytes([key[i * 4], key[i * 4 + 1], key[i * 4 + 2], key[i * 4 + 3]]);
        }

        state[12] = 0;

        for i in 0..3 {
            state[13 + i] = u32::from_le_bytes([
                nonce[i * 4],
                nonce[i * 4 + 1],
                nonce[i * 4 + 2],
                nonce[i * 4 + 3],
            ]);
        }

        let mut instance = Self {
            state,
            keystream: [0; 64],
            position: 64,
        };

        instance.generate_block();
        instance
    }

    pub fn new_with_two_seeds(seed1: u64, seed2: u64) -> Self {
        let mut key = [0u8; 32];
        let mut nonce = [0u8; 12];

        let s1_bytes = seed1.to_le_bytes();
        let s2_bytes = seed2.to_le_bytes();

        for i in 0..16 {
            key[i] = s1_bytes[i % 8];
            key[i + 16] = s2_bytes[i % 8];
        }

        let combined = seed1 ^ seed2;
        nonce[0..8].copy_from_slice(&combined.to_le_bytes());
        nonce[8..12]
            .copy_from_slice(&(combined.wrapping_mul(0x9e3779b97f4a7c15)).to_le_bytes()[0..4]);

        Self::new_with_key_and_nonce(&key, &nonce)
    }

    fn generate_block(&mut self) {
        let mut working_state = self.state;

        for _ in 0..4 {
            Self::quarter_round(0, 4, 8, 12, &mut working_state);
            Self::quarter_round(1, 5, 9, 13, &mut working_state);
            Self::quarter_round(2, 6, 10, 14, &mut working_state);
            Self::quarter_round(3, 7, 11, 15, &mut working_state);

            Self::quarter_round(0, 5, 10, 15, &mut working_state);
            Self::quarter_round(1, 6, 11, 12, &mut working_state);
            Self::quarter_round(2, 7, 8, 13, &mut working_state);
            Self::quarter_round(3, 4, 9, 14, &mut working_state);
        }

        for i in 0..16 {
            working_state[i] = working_state[i].wrapping_add(self.state[i]);
        }

        for i in 0..16 {
            let bytes = working_state[i].to_le_bytes();
            let offset = i * 4;
            self.keystream[offset] = bytes[0];
            self.keystream[offset + 1] = bytes[1];
            self.keystream[offset + 2] = bytes[2];
            self.keystream[offset + 3] = bytes[3];
        }

        let (new_counter, overflow) = self.state[12].overflowing_add(1);
        self.state[12] = new_counter;
        if overflow {
            self.state[13] = self.state[13].wrapping_add(1);
        }

        self.position = 0;
    }

    #[inline(always)]
    fn quarter_round(a: usize, b: usize, c: usize, d: usize, state: &mut [u32; 16]) {
        state[a] = state[a].wrapping_add(state[b]);
        state[d] = (state[d] ^ state[a]).rotate_left(16);
        state[c] = state[c].wrapping_add(state[d]);
        state[b] = (state[b] ^ state[c]).rotate_left(12);
        state[a] = state[a].wrapping_add(state[b]);
        state[d] = (state[d] ^ state[a]).rotate_left(8);
        state[c] = state[c].wrapping_add(state[d]);
        state[b] = (state[b] ^ state[c]).rotate_left(7);
    }

    pub fn next(&mut self) -> u64 {
        if self.position > 56 {
            self.generate_block();
        }

        let result = u64::from_le_bytes([
            self.keystream[self.position],
            self.keystream[self.position + 1],
            self.keystream[self.position + 2],
            self.keystream[self.position + 3],
            self.keystream[self.position + 4],
            self.keystream[self.position + 5],
            self.keystream[self.position + 6],
            self.keystream[self.position + 7],
        ]);

        self.position += 8;
        result
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.position > 60 {
            self.generate_block();
        }

        let result = u32::from_le_bytes([
            self.keystream[self.position],
            self.keystream[self.position + 1],
            self.keystream[self.position + 2],
            self.keystream[self.position + 3],
        ]);

        self.position += 4;
        result
    }

    pub fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut dest_pos = 0;

        while dest_pos < dest.len() {
            if self.position >= 64 {
                self.generate_block();
            }

            let available = 64 - self.position;
            let needed = dest.len() - dest_pos;
            let to_copy = available.min(needed);

            dest[dest_pos..dest_pos + to_copy]
                .copy_from_slice(&self.keystream[self.position..self.position + to_copy]);

            self.position += to_copy;
            dest_pos += to_copy;
        }
    }

    pub fn next_bool(&mut self, probability: f64) -> bool {
        if probability <= 0.0 {
            return false;
        }
        if probability >= 1.0 {
            return true;
        }

        let threshold = (probability * (u32::MAX as f64)) as u32;
        self.next_u32() < threshold
    }

    pub fn next_f64(&mut self) -> f64 {
        let bits = self.next() >> 11;
        (bits as f64) * (1.0 / (1u64 << 53) as f64)
    }

    pub fn next_f32(&mut self) -> f32 {
        let bits = self.next_u32() >> 8;
        (bits as f32) * (1.0 / (1u32 << 24) as f32)
    }

    pub fn stream_position(&self) -> usize {
        self.position + (self.state[12] as usize) * 64
    }

    pub fn reset(&mut self) {
        self.state[12] = 0;
        self.position = 64;
        self.generate_block();
    }

    pub fn reseed(&mut self, seed: u64) {
        *self = Self::new(seed);
    }

    pub fn reseed_with_two(&mut self, seed1: u64, seed2: u64) {
        *self = Self::new_with_two_seeds(seed1, seed2);
    }
}

impl Iterator for RandGenerator {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.next())
    }
}
