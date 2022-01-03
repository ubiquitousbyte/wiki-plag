package crawler

import (
	"errors"
	"reflect"
	"testing"

	"github.com/ubiquitousbyte/wiki-documents/entity"
	"github.com/ubiquitousbyte/wiki-documents/mediawiki"
)

func TestParseDocument(t *testing.T) {
	tests := []struct {
		name string
		page mediawiki.Page
		doc  entity.Document
		err  error
	}{
		{
			name: "parse page 1",
			page: mediawiki.Page{
				Id:    1,
				Title: "Page 1",
				Text:  "text\n==Par1==\ntext{\\randomgibberish}\n==Par2==text",
			},
			doc: entity.Document{
				Title:  "Page 1",
				Source: "mediawiki",
				Paragraphs: []entity.Paragraph{
					{
						Title:    "Abstract",
						Position: 1,
						Text:     "text",
					},
					{
						Title:    "Par1",
						Position: 2,
						Text:     "text",
					},
					{
						Title:    "Par2",
						Position: 2,
						Text:     "text",
					},
				},
			},
		},
		{
			name: "parse page 2",
			page: mediawiki.Page{
				Id:    2,
				Title: "Page 2",
				Text:  "text\n==Par1==\ntexttexttexttext{nosuchthing}===Par2===",
			},
			doc: entity.Document{
				Title:  "Page 2",
				Source: "mediawiki",
				Paragraphs: []entity.Paragraph{
					{
						Title:    "Abstract",
						Position: 1,
						Text:     "text",
					},
					{
						Title:    "Par1",
						Position: 2,
						Text:     "texttexttexttext",
					},
				},
			},
		},
		{
			name: "parse page 3",
			page: mediawiki.Page{
				Id:    3,
				Title: "Page 3",
				Text:  "==Title never ends",
			},
			err: ErrParse,
		},
		{
			name: "parse page 4",
			page: mediawiki.Page{
				Id:    4,
				Title: "Page 4",
				Text:  "{LateX Block never ends",
			},
			err: ErrParse,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			doc, err := parseDocument(&test.page)
			if !errors.Is(err, test.err) {
				t.Errorf("Expected error %s, but got %s instead", test.err, err)
			}

			if !reflect.DeepEqual(doc, test.doc) {
				t.Errorf("Expected document %v, but got %v instead", test.doc, doc)
			}
		})
	}
}

func TestParseCategory(t *testing.T) {
	tests := []struct {
		name     string
		page     mediawiki.Page
		category entity.Category
		err      error
	}{
		{
			name: "parse valid category",
			page: mediawiki.Page{
				Title:     "Category:Category 1",
				Namespace: mediawiki.NamespaceCategory,
				Text:      "Some text",
			},
			category: entity.Category{
				Name:        "Category 1",
				Description: "Some text",
				Source:      "mediawiki",
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cat, err := parseCategory(&test.page)
			if !errors.Is(err, test.err) {
				t.Errorf("Expected error %s, but got %s", test.err, err)
			}

			if cat != test.category {
				t.Errorf("Expected category %s, but got %s", test.category, cat)
			}
		})
	}
}
