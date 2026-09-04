use leone_gguf::{MetadataArray, MetadataValue};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use thiserror::Error;
use unicode_general_category::{get_general_category, GeneralCategory};

/// The GGUF token type used by the GPT-2 vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

/// An error returned while loading or applying a GGUF tokenizer.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    #[error("GGUF tokenizer key {0:?} is missing")]
    Missing(String),
    #[error("GGUF tokenizer key {key:?} has the wrong type, expected {expected}")]
    WrongType { key: String, expected: &'static str },
    #[error("tokenizer model {0:?} is not supported; expected gpt2 with qwen2 or llama-bpe pre-tokenization")]
    Unsupported(String),
    #[error("tokenizer has {tokens} tokens but {types} token types")]
    TypeCount { tokens: usize, types: usize },
    #[error("tokenizer token count exceeds u32")]
    TooManyTokens,
    #[error("tokenizer merge count exceeds u32")]
    TooManyMerges,
    #[error("token type code {code} at token {token} is not supported")]
    InvalidTokenType { token: usize, code: i32 },
    #[error("merge {index} has no separator")]
    InvalidMerge { index: usize },
    #[error("special token id {id} is outside a vocabulary of {tokens} tokens")]
    SpecialOutOfRange { id: u64, tokens: usize },
    #[error("token id {id} is outside a vocabulary of {tokens} tokens")]
    TokenOutOfRange { id: u32, tokens: usize },
    #[error("BPE piece {0:?} is absent from the vocabulary")]
    MissingPiece(String),
    #[error("tokenizer token {0:?} is missing")]
    MissingToken(String),
    #[error("decoded token bytes are not valid UTF-8")]
    InvalidUtf8,
    #[error("token {id} contains a character outside the GPT-2 byte map")]
    InvalidBytePiece { id: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreTokenizer {
    Qwen2,
    Llama3,
}

/// A supported GPT-2 byte-level BPE loaded from GGUF metadata.
#[derive(Debug)]
pub struct Tokenizer {
    tokens: Vec<String>,
    token_types: Vec<TokenType>,
    token_to_id: HashMap<String, u32>,
    merges: HashMap<(String, String), u32>,
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8>,
    bos: Option<u32>,
    eos: Option<u32>,
    add_bos: bool,
    add_eos: bool,
    pre_tokenizer: PreTokenizer,
}

struct Vocabulary {
    tokens: Vec<String>,
    token_types: Vec<TokenType>,
    token_to_id: HashMap<String, u32>,
}

impl Tokenizer {
    /// Loads one checked GPT-2 vocabulary, split rule, merge set, and special-token policy.
    pub fn from_metadata(
        metadata: &std::collections::BTreeMap<String, MetadataValue>,
    ) -> Result<Self, TokenizerError> {
        let pre_tokenizer = load_pre_tokenizer(metadata)?;
        let Vocabulary {
            tokens,
            token_types,
            token_to_id,
        } = load_vocabulary(metadata)?;
        let merges = load_merges(metadata)?;
        let (byte_encoder, byte_decoder) = byte_maps();
        let (bos, eos, add_bos, add_eos) = load_special_tokens(metadata, tokens.len())?;
        Ok(Self {
            tokens,
            token_types,
            token_to_id,
            merges,
            byte_encoder,
            byte_decoder,
            bos,
            eos,
            add_bos,
            add_eos,
            pre_tokenizer,
        })
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub const fn bos_token(&self) -> Option<u32> {
        self.bos
    }

    pub const fn eos_token(&self) -> Option<u32> {
        self.eos
    }

    /// Encodes plain text and applies the GGUF BOS and EOS defaults.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let mut output = Vec::new();
        if self.add_bos {
            if let Some(bos) = self.bos {
                output.push(bos);
            }
        }
        self.encode_piece_into(text, &mut output)?;
        if self.add_eos {
            if let Some(eos) = self.eos {
                output.push(eos);
            }
        }
        Ok(output)
    }

    /// Encodes plain text without adding BOS or EOS tokens.
    ///
    /// Text that spells a special token remains plain text. Call [`Self::token_id`]
    /// to add a trusted template token.
    pub fn encode_piece(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let mut output = Vec::new();
        self.encode_piece_into(text, &mut output)?;
        Ok(output)
    }

    /// Returns the exact vocabulary ID for one trusted template token.
    pub fn token_id(&self, token: &str) -> Result<u32, TokenizerError> {
        self.token_to_id
            .get(token)
            .copied()
            .ok_or_else(|| TokenizerError::MissingToken(token.to_owned()))
    }

    fn encode_piece_into(&self, text: &str, output: &mut Vec<u32>) -> Result<(), TokenizerError> {
        let digit_limit = match self.pre_tokenizer {
            PreTokenizer::Qwen2 => 1,
            PreTokenizer::Llama3 => 3,
        };
        for piece in bpe_split(text, digit_limit) {
            let encoded: String = piece
                .as_bytes()
                .iter()
                .map(|byte| self.byte_encoder[usize::from(*byte)])
                .collect();
            for token in self.merge_word(&encoded) {
                let id = self
                    .token_to_id
                    .get(&token)
                    .copied()
                    .ok_or_else(|| TokenizerError::MissingPiece(token.clone()))?;
                output.push(id);
            }
        }
        Ok(())
    }

    /// Returns decoded bytes for one token.
    ///
    /// Control and unused tokens return no bytes.
    pub fn token_bytes(&self, id: u32) -> Result<Vec<u8>, TokenizerError> {
        let mut output = Vec::new();
        self.token_bytes_into(id, &mut output)?;
        Ok(output)
    }

    /// Writes decoded bytes for one token into a reusable buffer.
    ///
    /// Control and unused tokens write no bytes.
    pub fn token_bytes_into(&self, id: u32, output: &mut Vec<u8>) -> Result<(), TokenizerError> {
        let index = usize::try_from(id).map_err(|_| TokenizerError::TokenOutOfRange {
            id,
            tokens: self.tokens.len(),
        })?;
        let token = self
            .tokens
            .get(index)
            .ok_or(TokenizerError::TokenOutOfRange {
                id,
                tokens: self.tokens.len(),
            })?;
        if matches!(
            self.token_types[index],
            TokenType::Control | TokenType::Unused
        ) {
            output.clear();
            return Ok(());
        }
        output.clear();
        for character in token.chars() {
            output.push(
                self.byte_decoder
                    .get(&character)
                    .copied()
                    .ok_or(TokenizerError::InvalidBytePiece { id })?,
            );
        }
        Ok(())
    }

    /// Returns the largest decoded token size in bytes.
    pub fn max_token_bytes(&self) -> usize {
        self.tokens
            .iter()
            .zip(&self.token_types)
            .filter(|(_, kind)| !matches!(kind, TokenType::Control | TokenType::Unused))
            .map(|(token, _)| token.chars().count())
            .max()
            .unwrap_or(0)
    }

    /// Decodes tokens to UTF-8 text and omits control and unused tokens.
    pub fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError> {
        let mut bytes = Vec::new();
        for token in tokens {
            bytes.extend(self.token_bytes(*token)?);
        }
        String::from_utf8(bytes).map_err(|_| TokenizerError::InvalidUtf8)
    }

    fn merge_word(&self, word: &str) -> Vec<String> {
        let symbol_count = word.chars().count();
        let mut symbols: Vec<Symbol> = word
            .chars()
            .enumerate()
            .map(|(index, character)| Symbol {
                text: character.to_string(),
                previous: index.checked_sub(1),
                next: (index + 1 < symbol_count).then_some(index + 1),
                live: true,
            })
            .collect();
        let mut queue = BinaryHeap::new();
        for right in 1..symbols.len() {
            self.push_pair(&symbols, right - 1, right, &mut queue);
        }
        self.merge_pairs(&mut symbols, &mut queue);
        symbols
            .into_iter()
            .filter(|symbol| symbol.live)
            .map(|symbol| symbol.text)
            .collect()
    }

    fn merge_pairs(&self, symbols: &mut [Symbol], queue: &mut BinaryHeap<Pair>) {
        while let Some(pair) = queue.pop() {
            if !symbols[pair.left].live
                || !symbols[pair.right].live
                || symbols[pair.left].next != Some(pair.right)
            {
                continue;
            }
            let current_rank = self
                .merges
                .get(&(
                    symbols[pair.left].text.clone(),
                    symbols[pair.right].text.clone(),
                ))
                .copied();
            if current_rank != Some(pair.rank) {
                continue;
            }
            let right_text = symbols[pair.right].text.clone();
            symbols[pair.left].text.push_str(&right_text);
            symbols[pair.right].live = false;
            symbols[pair.left].next = symbols[pair.right].next;
            if let Some(next) = symbols[pair.right].next {
                symbols[next].previous = Some(pair.left);
            }
            if let Some(previous) = symbols[pair.left].previous {
                self.push_pair(symbols, previous, pair.left, queue);
            }
            if let Some(next) = symbols[pair.left].next {
                self.push_pair(symbols, pair.left, next, queue);
            }
        }
    }

    fn push_pair(
        &self,
        symbols: &[Symbol],
        left: usize,
        right: usize,
        queue: &mut BinaryHeap<Pair>,
    ) {
        if let Some(rank) = self
            .merges
            .get(&(symbols[left].text.clone(), symbols[right].text.clone()))
        {
            queue.push(Pair {
                rank: *rank,
                left,
                right,
            });
        }
    }
}

fn load_pre_tokenizer(
    metadata: &std::collections::BTreeMap<String, MetadataValue>,
) -> Result<PreTokenizer, TokenizerError> {
    let model = required_string(metadata, "tokenizer.ggml.model")?;
    let pre = required_string(metadata, "tokenizer.ggml.pre")?;
    match (model, pre) {
        ("gpt2", "qwen2") => Ok(PreTokenizer::Qwen2),
        ("gpt2", "llama-bpe") => Ok(PreTokenizer::Llama3),
        _ => Err(TokenizerError::Unsupported(format!("{model}/{pre}"))),
    }
}

fn load_vocabulary(
    metadata: &std::collections::BTreeMap<String, MetadataValue>,
) -> Result<Vocabulary, TokenizerError> {
    let tokens = required_strings(metadata, "tokenizer.ggml.tokens")?.to_vec();
    let type_codes = required_i32(metadata, "tokenizer.ggml.token_type")?;
    if tokens.len() != type_codes.len() {
        return Err(TokenizerError::TypeCount {
            tokens: tokens.len(),
            types: type_codes.len(),
        });
    }
    u32::try_from(tokens.len()).map_err(|_| TokenizerError::TooManyTokens)?;
    let token_types = load_token_types(type_codes)?;
    let token_to_id = load_token_ids(&tokens)?;
    Ok(Vocabulary {
        tokens,
        token_types,
        token_to_id,
    })
}

fn load_token_types(type_codes: &[i32]) -> Result<Vec<TokenType>, TokenizerError> {
    type_codes
        .iter()
        .copied()
        .enumerate()
        .map(|(token, code)| token_type(token, code))
        .collect()
}

fn load_token_ids(tokens: &[String]) -> Result<HashMap<String, u32>, TokenizerError> {
    tokens
        .iter()
        .enumerate()
        .map(|(id, token)| {
            u32::try_from(id)
                .map(|id| (token.clone(), id))
                .map_err(|_| TokenizerError::TooManyTokens)
        })
        .collect()
}

fn load_merges(
    metadata: &std::collections::BTreeMap<String, MetadataValue>,
) -> Result<HashMap<(String, String), u32>, TokenizerError> {
    let mut merges = HashMap::new();
    for (index, merge) in required_strings(metadata, "tokenizer.ggml.merges")?
        .iter()
        .enumerate()
    {
        let (left, right) = merge
            .split_once(' ')
            .ok_or(TokenizerError::InvalidMerge { index })?;
        let rank = u32::try_from(index).map_err(|_| TokenizerError::TooManyMerges)?;
        merges.insert((left.to_owned(), right.to_owned()), rank);
    }
    Ok(merges)
}

fn load_special_tokens(
    metadata: &std::collections::BTreeMap<String, MetadataValue>,
    token_count: usize,
) -> Result<(Option<u32>, Option<u32>, bool, bool), TokenizerError> {
    let bos = optional_id(metadata, "tokenizer.ggml.bos_token_id", token_count)?;
    let eos = optional_id(metadata, "tokenizer.ggml.eos_token_id", token_count)?;
    let add_bos = optional_bool(metadata, "tokenizer.ggml.add_bos_token")?.unwrap_or(false);
    let add_eos = optional_bool(metadata, "tokenizer.ggml.add_eos_token")?.unwrap_or(false);
    Ok((bos, eos, add_bos, add_eos))
}

#[derive(Debug)]
struct Symbol {
    text: String,
    previous: Option<usize>,
    next: Option<usize>,
    live: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Pair {
    rank: u32,
    left: usize,
    right: usize,
}

impl Ord for Pair {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl PartialOrd for Pair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn bpe_split(text: &str, digit_limit: usize) -> Vec<String> {
    let characters: Vec<char> = text.chars().collect();
    let mut pieces = Vec::new();
    let mut position = 0;
    while position < characters.len() {
        pieces.push(bpe_piece(&characters, &mut position, digit_limit));
    }
    pieces
}

fn bpe_piece(characters: &[char], position: &mut usize, digit_limit: usize) -> String {
    let start = *position;
    if let Some(end) = contraction_end(characters, start) {
        *position = end;
        return characters[start..end].iter().collect();
    }
    if let Some(end) = letter_end(characters, start) {
        *position = end;
        return characters[start..end].iter().collect();
    }
    if let Some(end) = number_end(characters, start, digit_limit) {
        *position = end;
        return characters[start..end].iter().collect();
    }
    if let Some(end) = punctuation_end(characters, start) {
        *position = end;
        return characters[start..end].iter().collect();
    }
    let end = whitespace_end(characters, start);
    *position = end;
    characters[start..end].iter().collect()
}

fn contraction_end(characters: &[char], position: usize) -> Option<usize> {
    if characters[position] != '\'' || position + 1 >= characters.len() {
        return None;
    }
    let next = characters[position + 1].to_ascii_lowercase();
    if matches!(next, 's' | 't' | 'm' | 'd') {
        return Some(position + 2);
    }
    if position + 2 < characters.len() {
        let last = characters[position + 2].to_ascii_lowercase();
        if matches!((next, last), ('r', 'e') | ('v', 'e') | ('l', 'l')) {
            return Some(position + 3);
        }
    }
    None
}

fn letter_end(characters: &[char], position: usize) -> Option<usize> {
    let character = characters[position];
    if character == '\r'
        || character == '\n'
        || is_number(character)
        || (!is_letter(character) && !characters.get(position + 1).copied().is_some_and(is_letter))
    {
        return None;
    }
    let mut end = position + 1;
    while characters.get(end).copied().is_some_and(is_letter) {
        end += 1;
    }
    Some(end)
}

fn number_end(characters: &[char], position: usize, digit_limit: usize) -> Option<usize> {
    if !is_number(characters[position]) {
        return None;
    }
    let mut end = position + 1;
    while end - position < digit_limit && characters.get(end).copied().is_some_and(is_number) {
        end += 1;
    }
    Some(end)
}

fn punctuation_end(characters: &[char], position: usize) -> Option<usize> {
    let start = if characters[position] == ' ' {
        position + 1
    } else {
        position
    };
    if !is_punctuation_at(characters, start) {
        return None;
    }
    let mut end = start;
    while characters
        .get(end)
        .copied()
        .is_some_and(is_punctuation_character)
    {
        end += 1;
    }
    while matches!(characters.get(end), Some('\r' | '\n')) {
        end += 1;
    }
    Some(end)
}

fn is_punctuation_at(characters: &[char], position: usize) -> bool {
    characters
        .get(position)
        .copied()
        .is_some_and(is_punctuation_character)
}

fn is_punctuation_character(value: char) -> bool {
    !is_whitespace(value) && !is_letter(value) && !is_number(value)
}

fn whitespace_end(characters: &[char], position: usize) -> usize {
    let mut end = position;
    let mut last_newline = None;
    while characters.get(end).copied().is_some_and(is_whitespace) {
        if matches!(characters[end], '\r' | '\n') {
            last_newline = Some(end + 1);
        }
        end += 1;
    }
    if let Some(newline) = last_newline {
        newline
    } else if end - position > 1 && end < characters.len() {
        end - 1
    } else if end > position {
        end
    } else {
        position + 1
    }
}

fn is_letter(character: char) -> bool {
    matches!(
        get_general_category(character),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
    )
}

fn is_number(character: char) -> bool {
    matches!(
        get_general_category(character),
        GeneralCategory::DecimalNumber
            | GeneralCategory::LetterNumber
            | GeneralCategory::OtherNumber
    )
}

fn is_whitespace(character: char) -> bool {
    character.is_whitespace()
}

fn byte_maps() -> ([char; 256], HashMap<char, u8>) {
    let mut encoder = ['\0'; 256];
    let mut assigned = [false; 256];
    for byte in (b'!'..=b'~').chain(0xa1..=0xac).chain(0xae..=0xff) {
        encoder[usize::from(byte)] = char::from(byte);
        assigned[usize::from(byte)] = true;
    }
    let mut extra = 0_u32;
    for byte in 0_u16..=255 {
        if !assigned[usize::from(byte)] {
            encoder[usize::from(byte)] =
                char::from_u32(256 + extra).expect("GPT-2 byte map is valid Unicode");
            extra += 1;
        }
    }
    let decoder = encoder
        .iter()
        .copied()
        .enumerate()
        .map(|(byte, character)| (character, byte as u8))
        .collect();
    (encoder, decoder)
}

fn token_type(token: usize, code: i32) -> Result<TokenType, TokenizerError> {
    match code {
        1 => Ok(TokenType::Normal),
        2 => Ok(TokenType::Unknown),
        3 => Ok(TokenType::Control),
        4 => Ok(TokenType::UserDefined),
        5 => Ok(TokenType::Unused),
        6 => Ok(TokenType::Byte),
        _ => Err(TokenizerError::InvalidTokenType { token, code }),
    }
}

fn required_string<'a>(
    metadata: &'a std::collections::BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<&'a str, TokenizerError> {
    metadata
        .get(key)
        .and_then(MetadataValue::as_str)
        .ok_or_else(|| match metadata.contains_key(key) {
            true => TokenizerError::WrongType {
                key: key.to_owned(),
                expected: "string",
            },
            false => TokenizerError::Missing(key.to_owned()),
        })
}

fn required_strings<'a>(
    metadata: &'a std::collections::BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<&'a [String], TokenizerError> {
    match metadata.get(key) {
        Some(MetadataValue::Array(MetadataArray::String(values))) => Ok(values),
        Some(_) => Err(TokenizerError::WrongType {
            key: key.to_owned(),
            expected: "string array",
        }),
        None => Err(TokenizerError::Missing(key.to_owned())),
    }
}

fn required_i32<'a>(
    metadata: &'a std::collections::BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<&'a [i32], TokenizerError> {
    match metadata.get(key) {
        Some(MetadataValue::Array(MetadataArray::Int32(values))) => Ok(values),
        Some(_) => Err(TokenizerError::WrongType {
            key: key.to_owned(),
            expected: "i32 array",
        }),
        None => Err(TokenizerError::Missing(key.to_owned())),
    }
}

fn optional_id(
    metadata: &std::collections::BTreeMap<String, MetadataValue>,
    key: &str,
    tokens: usize,
) -> Result<Option<u32>, TokenizerError> {
    let Some(value) = metadata.get(key) else {
        return Ok(None);
    };
    let id = value.as_u64().ok_or_else(|| TokenizerError::WrongType {
        key: key.to_owned(),
        expected: "unsigned integer",
    })?;
    let token_count =
        u64::try_from(tokens).map_err(|_| TokenizerError::SpecialOutOfRange { id, tokens })?;
    if id >= token_count {
        return Err(TokenizerError::SpecialOutOfRange { id, tokens });
    }
    let id = u32::try_from(id).map_err(|_| TokenizerError::SpecialOutOfRange { id, tokens })?;
    Ok(Some(id))
}

fn optional_bool(
    metadata: &std::collections::BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<bool>, TokenizerError> {
    match metadata.get(key) {
        Some(MetadataValue::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(TokenizerError::WrongType {
            key: key.to_owned(),
            expected: "boolean",
        }),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use leone_gguf::Gguf;
    use std::error::Error;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};

    #[test]
    fn qwen2_split_keeps_expected_boundaries() {
        assert_eq!(
            bpe_split("Hello,  world\n42's", 1),
            ["Hello", ",", " ", " world", "\n", "4", "2", "'s"]
        );
        assert_eq!(bpe_split("  x", 1), [" ", " x"]);
    }

    #[test]
    fn llama3_split_groups_at_most_three_digits() {
        assert_eq!(bpe_split("1234567", 3), ["123", "456", "7"]);
    }

    #[test]
    fn byte_map_round_trips_every_byte() {
        let (encoder, decoder) = byte_maps();
        for byte in 0_u16..=255 {
            assert_eq!(decoder[&encoder[usize::from(byte)]], byte as u8);
        }
    }

    #[test]
    #[ignore = "requires the Qwen3 model and llama-tokenize oracle"]
    fn qwen3_tokenizer_matches_llama_cpp_on_fixed_corpus() -> Result<(), Box<dyn Error>> {
        let model = model_path();
        let gguf = Gguf::open(&model)?;
        let tokenizer = Tokenizer::from_metadata(gguf.metadata())?;
        let oracle = oracle_path();
        let cases = vec![
            "".to_owned(),
            "a".to_owned(),
            "Hello, world!".to_owned(),
            "The quick brown fox jumps over 13 lazy dogs.".to_owned(),
            " leading".to_owned(),
            "trailing ".to_owned(),
            "  two   spaces    here".to_owned(),
            "tabs\tand\nnewlines\r\n".to_owned(),
            "don't I'LL we're".to_owned(),
            "1234567890".to_owned(),
            "naive café".to_owned(),
            "smørbrød på blåbærtur".to_owned(),
            "Hei verden".to_owned(),
            "中文测试".to_owned(),
            "日本語の文".to_owned(),
            "Привет, мир".to_owned(),
            "مرحبا بالعالم".to_owned(),
            "emoji: 😀🚀🧪".to_owned(),
            "a\n\n\n b".to_owned(),
            "leone ".repeat(512),
        ];
        for text in cases {
            let expected = oracle_tokens(&oracle, &model, &text)?;
            let actual = tokenizer.encode(&text)?;
            assert_eq!(actual, expected, "token mismatch for {text:?}");
            assert_eq!(tokenizer.decode(&actual)?, text);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires the llama-bpe vocabulary and llama-tokenize oracle"]
    fn llama3_tokenizer_matches_llama_cpp_on_fixed_corpus() -> Result<(), Box<dyn Error>> {
        let model = llama_model_path();
        let gguf = Gguf::open(&model)?;
        let tokenizer = Tokenizer::from_metadata(gguf.metadata())?;
        let oracle = oracle_path();
        let cases = [
            "",
            "Hello, world!",
            "1234567890",
            "Llama 3 groups 123 digits, then 456.",
            "don't I'LL we're",
            "  two   spaces    here",
            "tabs\tand\nnewlines\r\n",
            "smørbrød på blåbærtur",
            "中文测试",
            "emoji: 😀🚀🧪",
        ];
        for text in cases {
            let expected = oracle_tokens(&oracle, &model, text)?;
            let actual = tokenizer.encode_piece(text)?;
            assert_eq!(actual, expected, "token mismatch for {text:?}");
            assert_eq!(tokenizer.decode(&actual)?, text);
        }
        Ok(())
    }

    fn oracle_tokens(oracle: &Path, model: &Path, text: &str) -> Result<Vec<u32>, Box<dyn Error>> {
        let mut child = Command::new(oracle)
            .args(["-m", model.to_str().ok_or("non-UTF-8 model path")?])
            .args(["--stdin", "--ids", "--no-escape", "--no-bos"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        child
            .stdin
            .take()
            .ok_or("missing oracle stdin")?
            .write_all(text.as_bytes())?;
        let output = child.wait_with_output()?;
        if !output.status.success() {
            return Err(format!("llama-tokenize exited with {}", output.status).into());
        }
        let stdout = String::from_utf8(output.stdout)?;
        let ids = stdout.trim().trim_start_matches('[').trim_end_matches(']');
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        ids.split(',')
            .map(|value| value.trim().parse().map_err(Into::into))
            .collect()
    }

    fn model_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/Qwen3-8B-Q4_K_M.gguf")
    }

    fn llama_model_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../external/llama.cpp/models/ggml-vocab-llama-bpe.gguf")
    }

    /// Returns the path of the `llama-tokenize` oracle.
    ///
    /// The default is the pinned build that `scripts/fetch-llama-cpp.sh`
    /// produces. Set `LLAMA_TOKENIZE` to point at another build.
    fn oracle_path() -> PathBuf {
        std::env::var_os("LLAMA_TOKENIZE")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../external/llama.cpp/build/bin/llama-tokenize")
            })
    }
}
