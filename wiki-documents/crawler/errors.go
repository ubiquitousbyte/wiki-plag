package crawler

import "fmt"

type errCode int

const (
	codeParse errCode = iota + 1
	codeCrawl
)

func (e errCode) String() string {
	switch e {
	case codeParse:
		return "Parse error"
	case codeCrawl:
		return "Crawl error"
	default:
		return "Unknown error"
	}
}

var (
	errParse = &crawlErr{code: codeParse}
	errCrawl = &crawlErr{code: codeCrawl}
)

type crawlErr struct {
	code errCode
	err  error
}

func (e crawlErr) Error() string {
	var msg string
	if e.err != nil {
		msg = e.err.Error()
	}
	return fmt.Sprintf("<CrawlerErr:%d>: %s (%s)", e.code, e.code, msg)
}

func (e crawlErr) from(err error) error {
	el := e
	el.err = err
	return el
}

func (e crawlErr) Unwrap() error {
	return e.err
}

func (e crawlErr) Is(other error) bool {
	m, ok := other.(*crawlErr)
	if !ok {
		return false
	}

	return m.code == e.code
}
