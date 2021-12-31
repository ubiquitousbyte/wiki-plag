package crawler

import (
	"errors"
	"io"
	"log"
	"reflect"
	"sync"
	"testing"

	mw "github.com/ubiquitousbyte/wiki-documents/mediawiki"
	"github.com/ubiquitousbyte/wiki-documents/models"
)

// In-memory MediaWiki client serving as a mock for Crawler testing
type mockMwClient struct {
	pages []mw.Page
	err   error
}

func (m *mockMwClient) ReadPages(req *mw.PageRequest) ([]mw.Page, error) {
	return m.pages, m.err
}

func (m *mockMwClient) ReadCategory(req *mw.PageRequest) (*mw.Page, error) {
	return nil, nil
}

func TestWalk(t *testing.T) {
	tests := []struct {
		name       string
		cfg        Config
		documents  []models.Document
		categories []models.Category
		err        error
	}{
		{
			name: "walk returns streams of categories and documents",
			cfg: Config{
				Language: "de",
				MwC: &mockMwClient{
					pages: []mw.Page{
						{
							Id:        1,
							Namespace: mw.NamespaceCategory,
							Title:     "Category:Category 1",
							Text:      "Category 1",
						},
						{
							Id:        2,
							Namespace: mw.NamespaceMain,
							Title:     "Document 1",
							Text:      "Document 1",
						},
						{
							Id:        3,
							Namespace: mw.NamespaceCategory,
							Title:     "Category:Category 2",
							Text:      "Category 2",
						},
						{
							Id:        4,
							Namespace: mw.NamespaceMain,
							Title:     "Document 2",
							Text:      "Document 2",
						},
					},
					err: io.EOF,
				},
				Logger: log.Default(),
			},
			documents: []models.Document{
				{
					Title:  "Document 1",
					Source: "mediawiki",
					Paragraphs: []models.Paragraph{
						{
							Title:    "Abstract",
							Position: 1,
							Text:     "Document 1",
						},
					},
				},
				{
					Title:  "Document 2",
					Source: "mediawiki",
					Paragraphs: []models.Paragraph{
						{
							Title:    "Abstract",
							Position: 1,
							Text:     "Document 2",
						},
					},
				},
			},
			categories: []models.Category{
				{
					Source:      "mediawiki",
					Name:        "Category 1",
					Description: "Category 1",
				},
				{
					Source:      "mediawiki",
					Name:        "Category 2",
					Description: "Category 2",
				},
			},
		},
		{
			name: "walk returns empty streams when no pages available",
			cfg: Config{
				Language: "de",
				MwC:      &mockMwClient{err: io.EOF},
				Logger:   log.Default(),
			},
			categories: make([]models.Category, 0),
			documents:  make([]models.Document, 0),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			crawler := NewCrawler(test.cfg)
			categories, documents, err := crawler.Walk(&models.Category{Name: "dummy"})
			if !errors.Is(err, test.err) {
				t.Errorf("Expected error %s, but got %s instead", test.err, err)
			}
			var wg sync.WaitGroup

			wg.Add(1)
			c := make([]models.Category, 0)
			go func() {
				defer wg.Done()
				for category := range categories {
					c = append(c, category)
				}
			}()

			wg.Add(1)
			d := make([]models.Document, 0)
			go func() {
				defer wg.Done()
				for doc := range documents {
					d = append(d, doc)
				}
			}()

			wg.Wait()

			if !reflect.DeepEqual(c, test.categories) {
				t.Errorf("Expected different categories")
			}

			if !reflect.DeepEqual(d, test.documents) {
				t.Errorf("Expected different documents")
			}
		})
	}
}
