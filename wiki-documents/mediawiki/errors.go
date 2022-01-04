package mediawiki

import "fmt"

type errCode int

const (
	codeInvalidRequest errCode = iota + 1
	codeInvalidEndpoint
	codeProto
	codeDecode
	codeNoPages
)

func (e errCode) String() string {
	switch e {
	case codeInvalidRequest:
		return "Invalid request"
	case codeInvalidEndpoint:
		return "Invalid endpoint"
	case codeProto:
		return "Protocol error"
	case codeDecode:
		return "Decode error"
	case codeNoPages:
		return "No pages found"
	default:
		return "Unknown error"
	}
}

var (
	ErrInvalidRequest  = &mwErr{code: codeInvalidRequest}
	ErrInvalidEndpoint = &mwErr{code: codeInvalidEndpoint}
	ErrProto           = &mwErr{code: codeProto}
	ErrDecode          = &mwErr{code: codeDecode}
	ErrNoPages         = &mwErr{code: codeNoPages}
)

type mwErr struct {
	code errCode
	err  error
}

func (e mwErr) Error() string {
	var msg string
	if e.err != nil {
		msg = e.err.Error()
	}
	return fmt.Sprintf("<MediaWikiErr:%d>: %s (%s)", e.code, e.code, msg)
}

func (e mwErr) from(err error) error {
	el := e
	el.err = err
	return el
}

func (e mwErr) Unwrap() error {
	return e.err
}

func (e mwErr) Is(other error) bool {
	m, ok := other.(*mwErr)
	if !ok {
		return false
	}

	return m.code == e.code
}
