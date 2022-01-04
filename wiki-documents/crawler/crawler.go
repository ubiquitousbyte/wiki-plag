// Package crawler provides an interface for traversing the MediaWiki graph
// and transforming nodes into application-specific entities
package crawler

import (
	"fmt"
	"io"
	"log"

	"github.com/ubiquitousbyte/wiki-documents/entity"
	mw "github.com/ubiquitousbyte/wiki-documents/mediawiki"
)

// Crawler configuration provided by the caller
type Config struct {
	Language string      // The language subsystem to crawl
	MwC      mw.Client   // The MediaWiki client to use for data extraction
	Logger   *log.Logger // The logger to use for wriing status messages
}

// Crawler is an object that can traverse the MediaWiki category graph
type Crawler struct {
	Config
}

// Creates a new crawler from the crawler configuration
func NewCrawler(config Config) *Crawler {
	return &Crawler{Config: config}
}

// extract loads data from the MediaWiki APIs
// This function returns an asynchronous MediaWiki Page stream by issuing
// the request provided by the caller.
func (cw *Crawler) extract(req *mw.PageRequest) <-chan mw.Page {
	pages := make(chan mw.Page, 1)
	go func() {
		defer close(pages)
		for ok := true; ok; {
			batch, err := cw.MwC.ReadPages(req)
			if err == io.EOF {
				ok = false
			} else if err != nil {
				cw.Logger.Println(ErrCrawl.from(err))
				break
			}

			for _, page := range batch {
				pages <- page
			}
		}
	}()
	return pages
}

// The function transforms every page in the page stream into either a category
// or document.
func (cw *Crawler) transform(p <-chan mw.Page) (<-chan entity.Category, <-chan entity.Document) {
	categories := make(chan entity.Category, 1)
	documents := make(chan entity.Document, 1)

	go func() {
		defer close(categories)
		defer close(documents)

		for page := range p {
			switch page.Namespace {
			case mw.NamespaceCategory:
				category, err := parseCategory(&page)
				if err != nil {
					cw.Logger.Println(ErrCrawl.from(err))
				} else {
					categories <- category
				}
			case mw.NamespaceMain:
				document, err := parseDocument(&page)
				if err != nil {
					cw.Logger.Println(ErrCrawl.from(err))
				} else {
					documents <- document
				}
			default:
				err := fmt.Errorf("Unknown page type: %s\n.", page.Title)
				cw.Logger.Println(ErrCrawl.from(err))
			}
		}
	}()

	return categories, documents
}

// Walk visits a node in the tree, extracting all of its children nodes.
// Children nodes are separated into categories and documents.
// The function returns two bounded streams of categories and documents, respectively.
func (cw *Crawler) Walk(node *entity.Category) (<-chan entity.Category, <-chan entity.Document, error) {
	request, err := mw.NewPageRequest(cw.Language, node.Name)
	if err != nil {
		return nil, nil, ErrCrawl.from(err)
	}
	pages := cw.extract(request)
	categories, documents := cw.transform(pages)
	return categories, documents, nil
}
