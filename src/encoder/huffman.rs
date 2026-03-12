use crate::encoder::bit_writer::BitWriter;
use crate::encoder::EncoderError;

const MAX_ALLOWED_CODE_LENGTH: usize = 15;

#[derive(Debug, Clone, Copy)]
struct HuffmanTreeNode {
    total_count: u32,
    value: isize,
    left: isize,
    right: isize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct HuffmanTreeToken {
    pub(crate) code: u8,
    pub(crate) extra_bits: u8,
}

#[derive(Debug, Clone)]
pub(crate) struct HuffmanCode {
    code_lengths: Vec<u8>,
    codes: Vec<u16>,
    single_symbol: Option<usize>,
}

impl HuffmanCode {
    pub(crate) fn from_code_lengths(code_lengths: Vec<u8>) -> Result<Self, EncoderError> {
        let mut counts = [0u32; MAX_ALLOWED_CODE_LENGTH + 1];
        let symbols = code_lengths
            .iter()
            .enumerate()
            .filter_map(|(symbol, &len)| (len != 0).then_some(symbol))
            .collect::<Vec<_>>();

        if symbols.is_empty() {
            return Err(EncoderError::Bitstream("empty Huffman tree"));
        }

        for &len in &code_lengths {
            let bits = len as usize;
            if bits > MAX_ALLOWED_CODE_LENGTH {
                return Err(EncoderError::Bitstream("invalid Huffman code length"));
            }
            if bits > 0 {
                counts[bits] += 1;
            }
        }

        let single_symbol = (symbols.len() == 1).then_some(symbols[0]);
        if symbols.len() > 1 {
            let mut left = 1i32;
            for bits in 1..=MAX_ALLOWED_CODE_LENGTH {
                left = (left << 1) - counts[bits] as i32;
                if left < 0 {
                    return Err(EncoderError::Bitstream("oversubscribed Huffman tree"));
                }
            }
            if left != 0 {
                return Err(EncoderError::Bitstream("incomplete Huffman tree"));
            }
        }

        let mut next_code = [0u32; MAX_ALLOWED_CODE_LENGTH + 1];
        let mut code = 0u32;
        for bits in 1..=MAX_ALLOWED_CODE_LENGTH {
            code = (code + counts[bits - 1]) << 1;
            next_code[bits] = code;
        }

        let mut codes = vec![0u16; code_lengths.len()];
        for (symbol, &len) in code_lengths.iter().enumerate() {
            let bits = len as usize;
            if bits == 0 {
                continue;
            }
            let canonical = next_code[bits];
            next_code[bits] += 1;
            codes[symbol] = reverse_bits(canonical, bits);
        }

        Ok(Self {
            code_lengths,
            codes,
            single_symbol,
        })
    }

    pub(crate) fn from_histogram(
        histogram: &[u32],
        tree_depth_limit: usize,
    ) -> Result<Self, EncoderError> {
        let code_lengths = generate_code_lengths(histogram, tree_depth_limit)?;
        Self::from_code_lengths(code_lengths)
    }

    pub(crate) fn code_lengths(&self) -> &[u8] {
        &self.code_lengths
    }

    pub(crate) fn used_symbols(&self) -> Vec<usize> {
        self.code_lengths
            .iter()
            .enumerate()
            .filter_map(|(symbol, &len)| (len != 0).then_some(symbol))
            .collect()
    }

    pub(crate) fn write_symbol(
        &self,
        bw: &mut BitWriter,
        symbol: usize,
    ) -> Result<(), EncoderError> {
        if let Some(single_symbol) = self.single_symbol {
            if symbol != single_symbol {
                return Err(EncoderError::Bitstream(
                    "attempted to write unexpected single-symbol Huffman code",
                ));
            }
            return Ok(());
        }

        let depth = *self
            .code_lengths
            .get(symbol)
            .ok_or(EncoderError::InvalidParam("Huffman symbol is out of range"))?
            as usize;
        if depth == 0 {
            return Err(EncoderError::Bitstream(
                "attempted to write unused Huffman symbol",
            ));
        }
        bw.put_bits(self.codes[symbol] as u32, depth)
    }
}

pub(crate) fn compress_huffman_tree(code_lengths: &[u8]) -> Vec<HuffmanTreeToken> {
    let mut tokens = Vec::with_capacity(code_lengths.len());
    let mut prev_value = 8u8;
    let mut index = 0usize;

    while index < code_lengths.len() {
        let value = code_lengths[index];
        let mut next = index + 1;
        while next < code_lengths.len() && code_lengths[next] == value {
            next += 1;
        }
        let runs = next - index;
        if value == 0 {
            code_repeated_zeros(runs, &mut tokens);
        } else {
            code_repeated_values(runs, value, prev_value, &mut tokens);
            prev_value = value;
        }
        index = next;
    }

    tokens
}

fn code_repeated_values(
    mut repetitions: usize,
    value: u8,
    prev_value: u8,
    tokens: &mut Vec<HuffmanTreeToken>,
) {
    if value != prev_value {
        tokens.push(HuffmanTreeToken {
            code: value,
            extra_bits: 0,
        });
        repetitions -= 1;
    }

    while repetitions >= 1 {
        if repetitions < 3 {
            for _ in 0..repetitions {
                tokens.push(HuffmanTreeToken {
                    code: value,
                    extra_bits: 0,
                });
            }
            break;
        } else if repetitions < 7 {
            tokens.push(HuffmanTreeToken {
                code: 16,
                extra_bits: (repetitions - 3) as u8,
            });
            break;
        } else {
            tokens.push(HuffmanTreeToken {
                code: 16,
                extra_bits: 3,
            });
            repetitions -= 6;
        }
    }
}

fn code_repeated_zeros(mut repetitions: usize, tokens: &mut Vec<HuffmanTreeToken>) {
    while repetitions >= 1 {
        if repetitions < 3 {
            for _ in 0..repetitions {
                tokens.push(HuffmanTreeToken {
                    code: 0,
                    extra_bits: 0,
                });
            }
            break;
        } else if repetitions < 11 {
            tokens.push(HuffmanTreeToken {
                code: 17,
                extra_bits: (repetitions - 3) as u8,
            });
            break;
        } else if repetitions < 139 {
            tokens.push(HuffmanTreeToken {
                code: 18,
                extra_bits: (repetitions - 11) as u8,
            });
            break;
        } else {
            tokens.push(HuffmanTreeToken {
                code: 18,
                extra_bits: 0x7f,
            });
            repetitions -= 138;
        }
    }
}

fn generate_code_lengths(
    histogram: &[u32],
    tree_depth_limit: usize,
) -> Result<Vec<u8>, EncoderError> {
    let mut code_lengths = vec![0u8; histogram.len()];
    let tree_size_orig = histogram.iter().filter(|&&count| count != 0).count();
    if tree_size_orig == 0 {
        return Err(EncoderError::Bitstream("empty Huffman histogram"));
    }
    if tree_size_orig > (1usize << (tree_depth_limit - 1)) {
        return Err(EncoderError::Bitstream("Huffman tree exceeds depth limit"));
    }

    let mut count_min = 1u32;
    loop {
        code_lengths.fill(0);
        let mut tree = histogram
            .iter()
            .enumerate()
            .filter_map(|(value, &count)| {
                (count != 0).then_some(HuffmanTreeNode {
                    total_count: count.max(count_min),
                    value: value as isize,
                    left: -1,
                    right: -1,
                })
            })
            .collect::<Vec<_>>();
        tree.sort_by(|a, b| {
            b.total_count
                .cmp(&a.total_count)
                .then_with(|| a.value.cmp(&b.value))
        });

        if tree.len() == 1 {
            code_lengths[tree[0].value as usize] = 1;
        } else {
            let mut tree_pool = Vec::with_capacity(tree.len() * 2);
            let mut tree_size = tree.len();
            while tree_size > 1 {
                tree_pool.push(tree[tree_size - 1]);
                tree_pool.push(tree[tree_size - 2]);
                let count = tree_pool[tree_pool.len() - 1].total_count
                    + tree_pool[tree_pool.len() - 2].total_count;
                tree_size -= 2;

                let mut insert_at = 0usize;
                while insert_at < tree_size && tree[insert_at].total_count > count {
                    insert_at += 1;
                }
                let new_node = HuffmanTreeNode {
                    total_count: count,
                    value: -1,
                    left: (tree_pool.len() - 1) as isize,
                    right: (tree_pool.len() - 2) as isize,
                };
                tree.insert(insert_at, new_node);
                tree_size += 1;
            }
            set_bit_depths(&tree[0], &tree_pool, &mut code_lengths, 0);
        }

        let max_depth = code_lengths.iter().copied().max().unwrap_or(0) as usize;
        if max_depth <= tree_depth_limit {
            return Ok(code_lengths);
        }

        count_min = count_min
            .checked_mul(2)
            .ok_or(EncoderError::Bitstream("Huffman count limit overflow"))?;
    }
}

fn set_bit_depths(
    node: &HuffmanTreeNode,
    pool: &[HuffmanTreeNode],
    bit_depths: &mut [u8],
    level: u8,
) {
    if node.left >= 0 {
        set_bit_depths(&pool[node.left as usize], pool, bit_depths, level + 1);
        set_bit_depths(&pool[node.right as usize], pool, bit_depths, level + 1);
    } else {
        bit_depths[node.value as usize] = level;
    }
}

fn reverse_bits(mut code: u32, bits: usize) -> u16 {
    let mut out = 0u32;
    for _ in 0..bits {
        out = (out << 1) | (code & 1);
        code >>= 1;
    }
    out as u16
}
