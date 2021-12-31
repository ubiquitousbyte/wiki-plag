package crawler

import (
	"errors"
	"strings"

	mw "github.com/ubiquitousbyte/wiki-documents/mediawiki"
	obj "github.com/ubiquitousbyte/wiki-documents/models"
)

// A convenience wrapper over a slice of paragraphs
type paragraphs []obj.Paragraph

const dataSource = "mediawiki"

// Returns a reference to the last paragraph, if one exists
// Otherwise, the function returns nil
func (p paragraphs) Last() *obj.Paragraph {
	if len(p) > 0 {
		return &p[len(p)-1]
	}
	return nil
}

// Checks whether the collection of paragraphs is empty
func (p paragraphs) IsEmpty() bool {
	return len(p) == 0
}

// parsePosition parses the position of a paragraph inside the text
func parsePosition(pos string) int {
	return int(strings.Count(pos, "=") / 2)
}

// parseTitle parses the title of a paragraph
func parseTitle(title string) string {
	return strings.TrimSpace(strings.ReplaceAll(title, "=", ""))
}

// parseDocument parses a Page into a Document
func parseDocument(page *mw.Page) (doc obj.Document, err error) {
	ps := paragraphs(make([]obj.Paragraph, 0))

	flushFunc := func(builder *strings.Builder) {
		if builder.Len() > 0 {
			if last := ps.Last(); last != nil {
				last.Text = builder.String()
				builder.Reset()
			}
		}
	}

	var sb strings.Builder
	lexer := newLexer(page.Text)
	for t := lexer.next(); t.typ != tokenTypeEOF; t = lexer.next() {
		switch t.typ {
		case tokenTypeText:
			if ps.IsEmpty() {
				p := obj.Paragraph{Title: "Abstract", Text: t.data, Position: 1}
				ps = append(ps, p)
			} else {
				if c := strings.ReplaceAll(t.data, " ", ""); len(c) != 0 {
					sb.WriteString(t.data)
				}
			}
		case tokenTypeTitle:
			flushFunc(&sb)
			p := obj.Paragraph{
				Title:    parseTitle(t.data),
				Position: parsePosition(t.data),
			}
			ps = append(ps, p)
		case tokenTypeErr:
			return doc, errParse.from(errors.New(t.data))
		}
	}

	flushFunc(&sb)

	doc.Title = page.Title
	doc.Source = dataSource
	doc.Paragraphs = ps

	return
}

// parseCategory parses a Page into a Category
func parseCategory(page *mw.Page) (c obj.Category, err error) {
	substr := strings.SplitN(page.Title, ":", 2)

	c.Name = substr[1]
	c.Description = page.Text
	c.Source = dataSource

	return
}
