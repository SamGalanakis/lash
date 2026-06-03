use compact_str::CompactString;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TokenKind {
    Ident(CompactString),
    String(CompactString),
    Number(f64),
    LBrace,
    RBrace,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Colon,
    At,
    Question,
    Dot,
    Bang,
    Equal,
    DoubleEqual,
    BangEqual,
    AndAnd,
    OrOr,
    Pipe,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    If,
    Else,
    For,
    In,
    Await,
    Cancel,
    Submit,
    Print,
    Call,
    And,
    Or,
    Not,
    True,
    False,
    Null,
    Eof,
}

#[derive(Debug, Error, PartialEq)]
pub enum LexError {
    #[error("unexpected `{ch}`")]
    UnexpectedChar { ch: char, offset: usize },
    #[error("unterminated string")]
    UnterminatedString { offset: usize },
    #[error("invalid number `{lexeme}`")]
    InvalidNumber { lexeme: String, offset: usize },
}

impl LexError {
    pub fn offset(&self) -> usize {
        match self {
            Self::UnexpectedChar { offset, .. }
            | Self::UnterminatedString { offset }
            | Self::InvalidNumber { offset, .. } => *offset,
        }
    }
}

pub fn lex(source: &str) -> Result<Vec<Token>, LexError> {
    let mut lexer = Lexer {
        source,
        chars: source.char_indices().peekable(),
    };
    lexer.lex_all()
}

struct Lexer<'a> {
    source: &'a str,
    chars: std::iter::Peekable<std::str::CharIndices<'a>>,
}

impl<'a> Lexer<'a> {
    fn lex_all(&mut self) -> Result<Vec<Token>, LexError> {
        let mut tokens = Vec::with_capacity((self.source.len() / 4).max(8));
        while let Some((offset, ch)) = self.peek() {
            if ch.is_whitespace() || ch == ';' {
                self.bump();
                continue;
            }
            if ch == '#' {
                self.skip_comment();
                continue;
            }
            if ch == '/' && self.peek_second() == Some('/') {
                self.bump();
                self.bump();
                self.skip_comment();
                continue;
            }

            let token = match ch {
                '{' => self.single(TokenKind::LBrace),
                '}' => self.single(TokenKind::RBrace),
                '(' => self.single(TokenKind::LParen),
                ')' => self.single(TokenKind::RParen),
                '[' => self.single(TokenKind::LBracket),
                ']' => self.single(TokenKind::RBracket),
                ',' => self.single(TokenKind::Comma),
                ':' => self.single(TokenKind::Colon),
                '@' => self.single(TokenKind::At),
                '?' => self.single(TokenKind::Question),
                '.' => self.single(TokenKind::Dot),
                '+' => self.single(TokenKind::Plus),
                '-' => self.single(TokenKind::Minus),
                '*' => self.single(TokenKind::Star),
                '/' => self.single(TokenKind::Slash),
                '%' => self.single(TokenKind::Percent),
                '=' => self.double_or_single('=', TokenKind::DoubleEqual, TokenKind::Equal),
                '!' => self.double_or_single('=', TokenKind::BangEqual, TokenKind::Bang),
                '&' => self.required_double('&', TokenKind::AndAnd)?,
                '|' => self.double_or_single('|', TokenKind::OrOr, TokenKind::Pipe),
                '<' => self.double_or_single('=', TokenKind::LessEqual, TokenKind::Less),
                '>' => self.double_or_single('=', TokenKind::GreaterEqual, TokenKind::Greater),
                '"' if self.starts_with_at(offset, "\"\"\"") => {
                    self.triple_string(false, "\"\"\"")?
                }
                '"' => self.string()?,
                'r' if self.starts_with_at(offset, "r\"\"\"") => {
                    self.triple_string(true, "\"\"\"")?
                }
                'r' if self.starts_with_at(offset, "r'''") => self.triple_string(true, "'''")?,
                c if is_ident_start(c) => self.ident_or_keyword(),
                c if c.is_ascii_digit() => self.number()?,
                _ => return Err(LexError::UnexpectedChar { ch, offset }),
            };
            tokens.push(token);
        }

        let end = self.source.len();
        tokens.push(Token {
            kind: TokenKind::Eof,
            span: Span { start: end, end },
        });
        Ok(tokens)
    }

    fn single(&mut self, kind: TokenKind) -> Token {
        let (start, ch) = self.bump().expect("single token requires input");
        Token {
            kind,
            span: Span {
                start,
                end: start + ch.len_utf8(),
            },
        }
    }

    fn double_or_single(
        &mut self,
        second: char,
        double_kind: TokenKind,
        single_kind: TokenKind,
    ) -> Token {
        let (start, ch) = self.bump().expect("double token requires input");
        let end = if self.consume_if(second) {
            start + ch.len_utf8() + second.len_utf8()
        } else {
            start + ch.len_utf8()
        };
        Token {
            kind: if end > start + ch.len_utf8() {
                double_kind
            } else {
                single_kind
            },
            span: Span { start, end },
        }
    }

    fn string(&mut self) -> Result<Token, LexError> {
        let (start, _) = self.bump().expect("string requires quote");
        let content_start = start + 1;
        let mut value: Option<String> = None;
        while let Some((offset, ch)) = self.bump() {
            match ch {
                '"' => {
                    let value = match value {
                        Some(value) => CompactString::from(value),
                        None => CompactString::from(&self.source[content_start..offset]),
                    };
                    return Ok(Token {
                        kind: TokenKind::String(value),
                        span: Span {
                            start,
                            end: offset + 1,
                        },
                    });
                }
                '\\' => {
                    let value =
                        value.get_or_insert_with(|| self.source[content_start..offset].to_string());
                    let Some((_, escaped)) = self.bump() else {
                        return Err(LexError::UnterminatedString { offset: start });
                    };
                    let translated = match escaped {
                        '"' => '"',
                        '\\' => '\\',
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        other => other,
                    };
                    value.push(translated);
                }
                other => {
                    if let Some(value) = &mut value {
                        value.push(other);
                    }
                }
            }
        }
        Err(LexError::UnterminatedString { offset: start })
    }

    fn triple_string(&mut self, raw: bool, delimiter: &str) -> Result<Token, LexError> {
        let (start, _) = self.peek().expect("triple string requires input");
        let delimiter_start = start + usize::from(raw);
        let content_start = delimiter_start + delimiter.len();
        self.consume_until_byte(content_start);

        if raw {
            let Some(relative_end) = self.source[content_start..].find(delimiter) else {
                return Err(LexError::UnterminatedString { offset: start });
            };
            let content_end = content_start + relative_end;
            let end = content_end + delimiter.len();
            let value = CompactString::from(&self.source[content_start..content_end]);
            self.consume_until_byte(end);
            return Ok(Token {
                kind: TokenKind::String(value),
                span: Span { start, end },
            });
        }

        let mut value = String::new();
        while let Some((offset, ch)) = self.bump() {
            if self.starts_with_at(offset, delimiter) {
                self.consume_until_byte(offset + delimiter.len());
                return Ok(Token {
                    kind: TokenKind::String(value.into()),
                    span: Span {
                        start,
                        end: offset + delimiter.len(),
                    },
                });
            }
            if ch == '\\' {
                let Some((_, escaped)) = self.bump() else {
                    return Err(LexError::UnterminatedString { offset: start });
                };
                let translated = match escaped {
                    '"' => '"',
                    '\\' => '\\',
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    other => other,
                };
                value.push(translated);
            } else {
                value.push(ch);
            }
        }

        Err(LexError::UnterminatedString { offset: start })
    }

    fn required_double(&mut self, expected: char, kind: TokenKind) -> Result<Token, LexError> {
        let (start, ch) = self.bump().expect("double token requires input");
        if !self.consume_if(expected) {
            return Err(LexError::UnexpectedChar { ch, offset: start });
        }
        Ok(Token {
            kind,
            span: Span {
                start,
                end: start + ch.len_utf8() + expected.len_utf8(),
            },
        })
    }

    fn ident_or_keyword(&mut self) -> Token {
        let (start, _) = self.peek().expect("identifier requires input");
        let mut end = start;
        while let Some((offset, ch)) = self.peek() {
            if !is_ident_continue(ch) {
                break;
            }
            end = offset + ch.len_utf8();
            self.bump();
        }
        let text = &self.source[start..end];
        let kind = match text {
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "for" => TokenKind::For,
            "in" => TokenKind::In,
            "await" => TokenKind::Await,
            "cancel" => TokenKind::Cancel,
            "submit" => TokenKind::Submit,
            "print" => TokenKind::Print,
            "call" => TokenKind::Call,
            "and" => TokenKind::And,
            "or" => TokenKind::Or,
            "not" => TokenKind::Not,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            "null" => TokenKind::Null,
            _ => TokenKind::Ident(text.into()),
        };
        Token {
            kind,
            span: Span { start, end },
        }
    }

    fn number(&mut self) -> Result<Token, LexError> {
        let (start, _) = self.peek().expect("number requires input");
        let mut end = start;
        let mut seen_dot = false;
        while let Some((offset, ch)) = self.peek() {
            if ch == '.' && !seen_dot {
                seen_dot = true;
                end = offset + 1;
                self.bump();
                continue;
            }
            if !ch.is_ascii_digit() {
                break;
            }
            end = offset + ch.len_utf8();
            self.bump();
        }
        let lexeme = &self.source[start..end];
        let value = lexeme.parse::<f64>().map_err(|_| LexError::InvalidNumber {
            lexeme: lexeme.to_string(),
            offset: start,
        })?;
        Ok(Token {
            kind: TokenKind::Number(value),
            span: Span { start, end },
        })
    }

    fn skip_comment(&mut self) {
        while let Some((_, ch)) = self.bump() {
            if ch == '\n' {
                break;
            }
        }
    }

    fn bump(&mut self) -> Option<(usize, char)> {
        self.chars.next()
    }

    fn peek(&mut self) -> Option<(usize, char)> {
        self.chars.peek().copied()
    }

    fn peek_second(&self) -> Option<char> {
        let mut chars = self.chars.clone();
        chars.next()?;
        chars.next().map(|(_, ch)| ch)
    }

    fn starts_with_at(&self, offset: usize, needle: &str) -> bool {
        self.source[offset..].starts_with(needle)
    }

    fn consume_until_byte(&mut self, end: usize) {
        while let Some((offset, _)) = self.peek() {
            if offset >= end {
                break;
            }
            self.bump();
        }
    }

    fn consume_if(&mut self, expected: char) -> bool {
        match self.peek() {
            Some((_, ch)) if ch == expected => {
                self.bump();
                true
            }
            _ => false,
        }
    }
}

fn is_ident_start(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphabetic()
}

fn is_ident_continue(ch: char) -> bool {
    is_ident_start(ch) || ch.is_ascii_digit()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexes_all_token_classes_and_comments() {
        let tokens = lex(r#"
            # comment
            // comment
            if else for in await cancel submit print call and or not true false null start
            name _x a1 "hi\n\t\"\\\r\q" 12 3.5 { } ( ) [ ] , : @ ? . ! = == != && || | < <= > >= + - * / %
            "#)
        .expect("lexing should succeed");

        let kinds: Vec<_> = tokens.into_iter().map(|token| token.kind).collect();
        assert_eq!(
            kinds,
            vec![
                TokenKind::If,
                TokenKind::Else,
                TokenKind::For,
                TokenKind::In,
                TokenKind::Await,
                TokenKind::Cancel,
                TokenKind::Submit,
                TokenKind::Print,
                TokenKind::Call,
                TokenKind::And,
                TokenKind::Or,
                TokenKind::Not,
                TokenKind::True,
                TokenKind::False,
                TokenKind::Null,
                TokenKind::Ident("start".into()),
                TokenKind::Ident("name".into()),
                TokenKind::Ident("_x".into()),
                TokenKind::Ident("a1".into()),
                TokenKind::String("hi\n\t\"\\\rq".into()),
                TokenKind::Number(12.0),
                TokenKind::Number(3.5),
                TokenKind::LBrace,
                TokenKind::RBrace,
                TokenKind::LParen,
                TokenKind::RParen,
                TokenKind::LBracket,
                TokenKind::RBracket,
                TokenKind::Comma,
                TokenKind::Colon,
                TokenKind::At,
                TokenKind::Question,
                TokenKind::Dot,
                TokenKind::Bang,
                TokenKind::Equal,
                TokenKind::DoubleEqual,
                TokenKind::BangEqual,
                TokenKind::AndAnd,
                TokenKind::OrOr,
                TokenKind::Pipe,
                TokenKind::Less,
                TokenKind::LessEqual,
                TokenKind::Greater,
                TokenKind::GreaterEqual,
                TokenKind::Plus,
                TokenKind::Minus,
                TokenKind::Star,
                TokenKind::Slash,
                TokenKind::Percent,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn rejects_unexpected_characters() {
        let err = lex("`").expect_err("lexing should fail");
        assert_eq!(err, LexError::UnexpectedChar { ch: '`', offset: 0 });
    }

    #[test]
    fn lexes_multiline_and_raw_multiline_strings() {
        let tokens = lex(r####"
            normal = """line1\n"quoted"
line2"""
            raw = r"""*** Begin Patch
@@
\n { untouched }
*** End Patch"""
            "####)
        .expect("lexing should succeed");

        let strings: Vec<_> = tokens
            .into_iter()
            .filter_map(|token| match token.kind {
                TokenKind::String(value) => Some(value),
                _ => None,
            })
            .collect();
        assert_eq!(
            strings,
            vec![
                CompactString::from("line1\n\"quoted\"\nline2"),
                CompactString::from("*** Begin Patch\n@@\n\\n { untouched }\n*** End Patch"),
            ]
        );
    }

    #[test]
    fn lexes_raw_triple_single_quoted_strings() {
        let tokens = lex(r####"
            script = r'''python3 - <<'PY'
print("""double quotes are preserved""")
\n { braces stay raw }
PY'''
            "####)
        .expect("lexing should succeed");

        let strings: Vec<_> = tokens
            .into_iter()
            .filter_map(|token| match token.kind {
                TokenKind::String(value) => Some(value),
                _ => None,
            })
            .collect();
        assert_eq!(
            strings,
            vec![CompactString::from(
                "python3 - <<'PY'\nprint(\"\"\"double quotes are preserved\"\"\")\n\\n { braces stay raw }\nPY"
            )]
        );
    }

    #[test]
    fn lexes_label_annotation_text_inside_strings_as_strings() {
        let tokens = lex(r####"
            regular = "@label(title: \"plain\")"
            multiline = """@label(title: "plain")
finish null"""
            raw = r'''@label(title: "plain")
@label(title: "still plain") finish null'''
            "####)
        .expect("lexing should succeed");

        assert!(
            tokens
                .iter()
                .all(|token| !matches!(token.kind, TokenKind::At)),
            "`@` inside strings must not lex as annotation syntax"
        );
        let strings: Vec<_> = tokens
            .into_iter()
            .filter_map(|token| match token.kind {
                TokenKind::String(value) => Some(value),
                _ => None,
            })
            .collect();
        assert_eq!(
            strings,
            vec![
                CompactString::from("@label(title: \"plain\")"),
                CompactString::from("@label(title: \"plain\")\nfinish null"),
                CompactString::from(
                    "@label(title: \"plain\")\n@label(title: \"still plain\") finish null"
                ),
            ]
        );
    }

    #[test]
    fn lexes_double_slash_comments_without_breaking_division() {
        let tokens = lex(r#"
            value = 6 / 2
            // trailing comment
            submit value
            "#)
        .expect("lexing should succeed");

        let kinds: Vec<_> = tokens.into_iter().map(|token| token.kind).collect();
        assert_eq!(
            kinds,
            vec![
                TokenKind::Ident("value".into()),
                TokenKind::Equal,
                TokenKind::Number(6.0),
                TokenKind::Slash,
                TokenKind::Number(2.0),
                TokenKind::Submit,
                TokenKind::Ident("value".into()),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn rejects_unterminated_strings() {
        let err = lex("\"abc").expect_err("lexing should fail");
        assert_eq!(err, LexError::UnterminatedString { offset: 0 });

        let err = lex("\"abc\\").expect_err("lexing should fail");
        assert_eq!(err, LexError::UnterminatedString { offset: 0 });

        let err = lex("\"\"\"abc").expect_err("lexing should fail");
        assert_eq!(err, LexError::UnterminatedString { offset: 0 });

        let err = lex("r\"\"\"abc").expect_err("lexing should fail");
        assert_eq!(err, LexError::UnterminatedString { offset: 0 });

        let err = lex("r'''abc").expect_err("lexing should fail");
        assert_eq!(err, LexError::UnterminatedString { offset: 0 });
    }

    #[test]
    fn internal_number_error_path_is_covered() {
        let mut lexer = Lexer {
            source: ".",
            chars: ".".char_indices().peekable(),
        };
        let err = lexer.number().expect_err("number parsing should fail");
        assert_eq!(
            err,
            LexError::InvalidNumber {
                lexeme: ".".to_string(),
                offset: 0
            }
        );
    }

    #[test]
    fn identifier_helpers_cover_true_and_false_cases() {
        assert!(is_ident_start('_'));
        assert!(is_ident_start('a'));
        assert!(!is_ident_start('1'));

        assert!(is_ident_continue('9'));
        assert!(is_ident_continue('_'));
        assert!(!is_ident_continue('-'));
    }
}
