//! Token counting for tokenizers shaped like Anthropic's (optional NFKC
//! normalizer, `ByteLevel` pre-tokenizer with the GPT-2 split regex, no
//! post-processor) without running the regex. Oniguruma spends ~90% of
//! `encode_fast` on `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`;
//! a hand-written scanner finds the same pieces and hands them to the model.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::iter;

use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::{Model, Tokenizer};
use unicode_normalization_alignments::{IsNormalized, UnicodeNormalization, is_nfkc_quick};

use super::unicode_classes::{LETTER_RANGES, NUMBER_RANGES, SPACE_RANGES};

const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];

pub(super) struct ByteLevelCounter {
    nfkc: bool,
}

impl ByteLevelCounter {
    pub(super) fn detect(tokenizer: &Tokenizer) -> Option<Self> {
        let nfkc = match tokenizer.get_normalizer() {
            None => false,
            Some(NormalizerWrapper::NFKC(_)) => true,
            Some(_) => return None,
        };
        let Some(PreTokenizerWrapper::ByteLevel(byte_level)) = tokenizer.get_pre_tokenizer() else {
            return None;
        };
        let plain = !byte_level.add_prefix_space
            && byte_level.use_regex
            && tokenizer.get_post_processor().is_none()
            && tokenizer.get_truncation().is_none()
            && tokenizer.get_padding().is_none();
        plain.then_some(Self { nfkc })
    }

    /// `None` when the text contains an added token or the model rejects a
    /// piece; the caller then runs the full encoder.
    pub(super) fn count(&self, tokenizer: &Tokenizer, text: &str) -> Option<usize> {
        let normalized = self.normalize(text);
        let added_tokens = tokenizer.get_added_vocabulary().get_vocab();
        if added_tokens
            .keys()
            .any(|token| text.contains(token.as_str()) || normalized.contains(token.as_str()))
        {
            return None;
        }
        let model = tokenizer.get_model();
        let mapped: String = normalized.bytes().map(byte_char).collect();
        pieces(&normalized)
            .try_fold((0, 0), |(start, total), piece| {
                let end = start + mapped_len(piece);
                let tokens = model.tokenize(&mapped[start..end]).ok()?;
                Some((end, total + tokens.len()))
            })
            .map(|(_, total)| total)
    }

    /// Same crate and Unicode tables as `NormalizedString::nfkc`, so the
    /// result is what the full encoder would have tokenized.
    fn normalize<'a>(&self, text: &'a str) -> Cow<'a, str> {
        if !self.nfkc || text.is_ascii() || is_nfkc_quick(text.chars()) == IsNormalized::Yes {
            return Cow::Borrowed(text);
        }
        Cow::Owned(text.nfkc().map(|(character, _)| character).collect())
    }
}

/// GPT-2 `bytes_to_unicode`: printable Latin-1 bytes map to themselves, the
/// rest to U+0100 onwards in byte order.
fn byte_char(byte: u8) -> char {
    let code = match byte {
        0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF => u32::from(byte),
        0x00..=0x20 => 0x100 + u32::from(byte),
        0x7F..=0xA0 => 0x121 + u32::from(byte - 0x7F),
        0xAD => 0x143,
    };
    char::from_u32(code).unwrap_or(char::REPLACEMENT_CHARACTER)
}

fn mapped_len(piece: &str) -> usize {
    piece.len()
        + piece
            .bytes()
            .filter(|byte| !byte.is_ascii_graphic())
            .count()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    Letter,
    Number,
    Space,
    Other,
}

fn class(character: char) -> Class {
    match character {
        'A'..='Z' | 'a'..='z' => Class::Letter,
        '0'..='9' => Class::Number,
        '\t'..='\r' | ' ' => Class::Space,
        _ if character.is_ascii() => Class::Other,
        _ if in_ranges(LETTER_RANGES, character) => Class::Letter,
        _ if in_ranges(NUMBER_RANGES, character) => Class::Number,
        _ if in_ranges(SPACE_RANGES, character) => Class::Space,
        _ => Class::Other,
    }
}

fn in_ranges(ranges: &[(u32, u32)], character: char) -> bool {
    let code = u32::from(character);
    ranges
        .binary_search_by(|(low, high)| {
            if *high < code {
                Ordering::Less
            } else if *low > code {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        })
        .is_ok()
}

/// The regex matches every character, so the pieces tile the text.
fn pieces(text: &str) -> impl Iterator<Item = &str> {
    iter::successors(split_piece(text), |(_, rest)| split_piece(rest)).map(|(piece, _)| piece)
}

fn split_piece(text: &str) -> Option<(&str, &str)> {
    let first = text.chars().next()?;
    Some(text.split_at(piece_len(text, first)))
}

fn piece_len(text: &str, first: char) -> usize {
    if let Some(contraction) = CONTRACTIONS.iter().find(|word| text.starts_with(**word)) {
        return contraction.len();
    }
    let first_class = class(first);
    if first_class != Class::Space {
        return run_len(text, first_class);
    }
    if first != ' ' {
        return space_run_len(text);
    }
    let after_space = &text[1..];
    match after_space.chars().next().map(class) {
        None | Some(Class::Space) => space_run_len(text),
        Some(run_class) => 1 + run_len(after_space, run_class),
    }
}

fn run_len(text: &str, run_class: Class) -> usize {
    text.char_indices()
        .find(|(_, character)| class(*character) != run_class)
        .map_or(text.len(), |(index, _)| index)
}

/// `\s+(?!\S)|\s+`: whitespace followed by a non-space leaves its last
/// character to start the next piece (` ?` on the following alternatives).
fn space_run_len(text: &str) -> usize {
    let run = run_len(text, Class::Space);
    if run == text.len() {
        return run;
    }
    let last = text[..run].chars().next_back().map_or(0, char::len_utf8);
    match run - last {
        0 => run,
        shorter => shorter,
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::{Rng, SeedableRng};
    use tokenizers::normalizers::NFKC;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::utils::SysRegex;
    use tokenizers::{
        NormalizedString, Normalizer, OffsetReferential, OffsetType, PreTokenizedString,
        PreTokenizer,
    };

    use super::*;

    fn anthropic() -> Tokenizer {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../litellm/litellm_core_utils/tokenizers/anthropic_tokenizer.json"
        );
        std::fs::read_to_string(path)
            .expect("anthropic tokenizer json is in the repo")
            .parse()
            .expect("anthropic tokenizer loads")
    }

    fn reference_count(tokenizer: &Tokenizer, text: &str) -> usize {
        tokenizer.encode_fast(text, true).expect("encode").len()
    }

    const ALPHABET: &[&str] = &[
        "a",
        "Z",
        "e",
        "s",
        "t",
        "d",
        "m",
        "'",
        "'s",
        "'re",
        "'ll",
        "'S",
        "0",
        "9",
        " ",
        "  ",
        "\t",
        "\n",
        "\r\n",
        "\u{b}",
        ".",
        ",",
        "!",
        "-",
        "(",
        "\"",
        "\u{a0}",
        "\u{85}",
        "\u{2028}",
        "\u{3000}",
        "\u{200b}",
        "\u{200d}",
        "é",
        "e\u{301}",
        "ß",
        "漢",
        "字",
        "ع",
        "३",
        "½",
        "Ⅳ",
        "🙂",
        "👍🏽",
        "Ａ",
        "ﬁ",
        "㍿",
        "㋿",
        "ꟲ",
        "𐞁",
        "a\u{30a}",
        "\u{1e0b}\u{323}",
        "<",
        ">",
        "EOT",
        "<EOT>",
        "<META_START>",
    ];

    fn random_text(rng: &mut StdRng) -> String {
        let pieces = rng.gen_range(0..40);
        (0..pieces)
            .map(|_| *ALPHABET.choose(rng).expect("alphabet is not empty"))
            .collect()
    }

    #[test]
    fn anthropic_tokenizer_takes_the_fast_path() {
        let tokenizer = anthropic();
        let fast = ByteLevelCounter::detect(&tokenizer).expect("anthropic shape is supported");
        assert!(fast.nfkc);
        let text = "Hello, how are you today?";
        assert_eq!(
            fast.count(&tokenizer, text),
            Some(reference_count(&tokenizer, text))
        );
        assert_eq!(fast.count(&tokenizer, "stop <EOT> here"), None);
        assert_eq!(fast.count(&tokenizer, "stop ＜ＥＯＴ＞ here"), None);
    }

    #[test]
    fn counts_match_the_full_encoder() {
        let tokenizer = anthropic();
        let fast = ByteLevelCounter::detect(&tokenizer).expect("supported");
        let mut rng = StdRng::seed_from_u64(2026);
        for _ in 0..4000 {
            let text = random_text(&mut rng);
            let expected = reference_count(&tokenizer, &text);
            let counted = fast
                .count(&tokenizer, &text)
                .unwrap_or_else(|| reference_count(&tokenizer, &text));
            assert_eq!(counted, expected, "text {text:?}");
        }
    }

    #[test]
    fn nfkc_matches_the_tokenizer_normalizer_for_every_scalar_value() {
        let fast = ByteLevelCounter { nfkc: true };
        let mut text = String::new();
        for character in (0..=0x10FFFFu32).filter_map(char::from_u32) {
            text.clear();
            text.push(character);
            let mut expected = NormalizedString::from(text.as_str());
            NFKC.normalize(&mut expected).expect("nfkc");
            assert_eq!(
                fast.normalize(&text),
                expected.get(),
                "U+{:04X}",
                u32::from(character)
            );
        }
    }

    #[test]
    fn nfkc_matches_the_tokenizer_normalizer_on_random_texts() {
        let fast = ByteLevelCounter { nfkc: true };
        let mut rng = StdRng::seed_from_u64(11);
        for _ in 0..4000 {
            let text = random_text(&mut rng);
            let mut expected = NormalizedString::from(text.as_str());
            NFKC.normalize(&mut expected).expect("nfkc");
            assert_eq!(fast.normalize(&text), expected.get(), "text {text:?}");
        }
    }

    #[test]
    fn pieces_match_the_byte_level_pre_tokenizer() {
        let byte_level = ByteLevel::new(false, true, true);
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..4000 {
            let text = ByteLevelCounter { nfkc: true }
                .normalize(&random_text(&mut rng))
                .into_owned();
            let mut pre_tokenized = PreTokenizedString::from(text.as_str());
            byte_level
                .pre_tokenize(&mut pre_tokenized)
                .expect("pre-tokenize");
            let expected: Vec<(String, (usize, usize))> = pre_tokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(mapped, offsets, _)| (mapped.to_string(), offsets))
                .collect();
            let actual: Vec<(String, (usize, usize))> = pieces(&text)
                .map(|piece| {
                    let start = piece.as_ptr() as usize - text.as_ptr() as usize;
                    let mapped: String = piece.bytes().map(byte_char).collect();
                    (mapped, (start, start + piece.len()))
                })
                .collect();
            assert_eq!(actual, expected, "text {text:?}");
        }
    }

    #[test]
    fn byte_chars_match_the_byte_level_alphabet() {
        let byte_level = ByteLevel::new(false, false, false);
        let characters: Vec<char> = (0..=0x10FFFFu32).filter_map(char::from_u32).collect();
        for chunk in characters.chunks(1024) {
            let text: String = chunk.iter().collect();
            let mut pre_tokenized = PreTokenizedString::from(text.as_str());
            byte_level
                .pre_tokenize(&mut pre_tokenized)
                .expect("pre-tokenize");
            let expected: String = pre_tokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(mapped, _, _)| mapped)
                .collect();
            let actual: String = text.bytes().map(byte_char).collect();
            assert_eq!(actual.len(), mapped_len(&text));
            assert_eq!(
                actual,
                expected,
                "chunk starting at U+{:04X}",
                u32::from(chunk[0])
            );
        }
    }

    #[test]
    fn classes_match_oniguruma() {
        let letter = SysRegex::new(r"\p{L}").expect("regex");
        let number = SysRegex::new(r"\p{N}").expect("regex");
        let space = SysRegex::new(r"\s").expect("regex");
        let whole =
            |regex: &SysRegex, text: &str| regex.find_iter(text).next() == Some((0, text.len()));
        let mut text = String::new();
        for character in (0..=0x10FFFFu32).filter_map(char::from_u32) {
            text.clear();
            text.push(character);
            let expected = if whole(&letter, &text) {
                Class::Letter
            } else if whole(&number, &text) {
                Class::Number
            } else if whole(&space, &text) {
                Class::Space
            } else {
                Class::Other
            };
            assert_eq!(class(character), expected, "U+{:04X}", u32::from(character));
        }
    }

    #[test]
    fn other_tokenizer_shapes_are_declined() {
        let mut tokenizer = anthropic();
        tokenizer.with_pre_tokenizer(Some(ByteLevel::new(true, true, true)));
        assert!(ByteLevelCounter::detect(&tokenizer).is_none());
    }
}
