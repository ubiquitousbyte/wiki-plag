package crawler

import (
	"log"
	"sync"
	"testing"

	"github.com/ubiquitousbyte/wiki-documents/mediawiki"
	"github.com/ubiquitousbyte/wiki-documents/models"
)

func TestWalkMediaWiki(t *testing.T) {
	tests := []struct {
		name             string
		lang             string
		category         models.Category
		minDocCount      int
		minCategoryCount int
	}{
		{
			name: "walk Informatik page in german wikipedia",
			lang: "de",
			category: models.Category{
				Name: "Informatik",
			},
			minDocCount:      90, // See https://de.wikipedia.org/wiki/Kategorie:Informatik
			minCategoryCount: 11,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			crawler := NewCrawler(Config{
				Language: test.lang,
				MwC:      mediawiki.NewClient(),
				Logger:   log.Default(),
			})

			categories, docs, err := crawler.Walk(&test.category)
			if err != nil {
				t.Errorf("Unexpected error %s", err)
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
				for doc := range docs {
					d = append(d, doc)
				}
			}()

			wg.Wait()

			if len(d) < test.minDocCount {
				t.Errorf("Expected at least %d documents, but received %d", test.minDocCount, len(d))
			}

			if len(c) < test.minCategoryCount {
				t.Errorf("Expected at least %d categories, but received %d", test.minCategoryCount, len(c))
			}
		})
	}
}
