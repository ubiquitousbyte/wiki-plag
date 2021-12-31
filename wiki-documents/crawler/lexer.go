package crawler

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

type tokenType int

const (
	tokenTypeErr tokenType = iota
	tokenTypeText
	tokenTypeTitle
	tokenTypeEOF
)

const lexEOF = utf8.RuneError

const (
	tokenTitle      = "==" // Titles are denoted by multiple equal signs
	tokenLF         = "\n" // Line feeds
	tokenLatexBegin = "{"  // Begin of LateX code inside wikitext
	tokenLatexEnd   = "}"  // End of LateX code inside wikitext
)

type token struct {
	typ  tokenType
	data string
}

// Lexer state function
type lexerState func(l *lexer) lexerState

// A lexer that can interpret basic wikitext tokens
type lexer struct {
	input  string     // The string being scanned
	start  int        // The start position of the current token
	pos    int        // The current position in the input
	width  int        // The width in bytes of the last rune
	tokens chan token // Stream of tokens to emit
	state  lexerState // The current state of the lexer
}

// Creates a new lexer that will process the given input string
func newLexer(data string) *lexer {
	return &lexer{
		input:  data,
		tokens: make(chan token, 2),
		state:  lex,
	}
}

// Advances the lexer to the next utf8 character
func (l *lexer) nextRune() (r rune) {
	if l.pos > len(l.input) {
		l.width = 0
		return lexEOF
	}
	r, l.width = utf8.DecodeRuneInString(l.input[l.pos:])
	l.pos += l.width
	return r
}

// Yields the current token and advances the lexer to the next one
func (l *lexer) yield(t tokenType) {
	l.tokens <- token{t, l.input[l.start:l.pos]}
	l.start = l.pos
}

// Ignores the current token
func (l *lexer) ignore() {
	l.start = l.pos
}

// Goes back to the previous rune
func (l *lexer) backup() {
	l.pos -= l.width
}

// Peeks at the next rune
func (l *lexer) peek() rune {
	r := l.nextRune()
	l.backup()
	return r
}

func (l *lexer) error(format string, args ...interface{}) lexerState {
	l.tokens <- token{
		typ:  tokenTypeErr,
		data: fmt.Sprintf(format, args...),
	}
	return nil
}

// Lexes a title token
func lexTitle(l *lexer) lexerState {
	var lefties int
	var leftFinished bool
	var righties int
	for {
		r := l.nextRune()
		if r == lexEOF {
			if lefties != righties {
				return l.error("%s", "Invalid title token")
			}
			l.yield(tokenTypeEOF)
			return nil
		}
		if string(r) == "=" {
			if !leftFinished {
				lefties += 1
			} else {
				righties += 1
			}
		} else {
			if !leftFinished {
				leftFinished = true
			}
			if lefties == righties {
				l.backup()
				break
			}
		}
	}
	l.yield(tokenTypeTitle)
	return lex
}

// Lexes a line feed token
func lexLF(l *lexer) lexerState {
	if n := l.nextRune(); n == lexEOF {
		l.yield(tokenTypeEOF)
		return nil
	}
	l.ignore()
	return lex
}

// Lexes a LateX segment
func lexLatex(l *lexer) lexerState {
	var lefties int
	var righties int
	for {
		n := l.nextRune()
		if n == lexEOF {
			if lefties != righties {
				return l.error("%s", "Invalid LateX token")
			}
			l.yield(tokenTypeEOF)
			return nil
		}
		if string(n) == tokenLatexBegin {
			lefties += 1
		} else if string(n) == tokenLatexEnd {
			righties += 1
		}

		if lefties == righties {
			l.ignore()
			break
		}
	}
	return lex
}

// Main lexer loop
func lex(l *lexer) lexerState {
	for {
		if strings.HasPrefix(l.input[l.pos:], tokenLF) {
			if l.pos > l.start {
				l.yield(tokenTypeText)
			}
			return lexLF
		}

		if strings.HasPrefix(l.input[l.pos:], tokenTitle) {
			if l.pos > l.start {
				l.yield(tokenTypeText)
			}
			return lexTitle
		}

		if strings.HasPrefix(l.input[l.pos:], tokenLatexBegin) {
			if l.pos > l.start {
				l.yield(tokenTypeText)
			}
			return lexLatex
		}

		if l.nextRune() == lexEOF {
			break
		}
	}

	if l.pos > l.start {
		l.yield(tokenTypeText)
		return lex
	}

	l.yield(tokenTypeEOF)
	return nil
}

// Yields the next token
func (l *lexer) next() token {
	for {
		select {
		case tok := <-l.tokens:
			return tok
		default:
			if l.state = l.state(l); l.state == nil {
				close(l.tokens)
			}
		}
	}
}
