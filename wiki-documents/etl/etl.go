// Package etl provides an Extract-Transform-Load interface
// for extracting documents from MediaWiki, transforming them into
// application-specific entities (Category and Document), and loading
// them into a data repository.
package etl

import (
	"errors"

	"github.com/ubiquitousbyte/wiki-documents/crawler"
	db "github.com/ubiquitousbyte/wiki-documents/database"
	obj "github.com/ubiquitousbyte/wiki-documents/models"
)

type Config struct {
	crawler.Config
	Ds    db.DocumentStore
	Cs    db.CategoryStore
	Depth uint8
}

type Pipeline struct {
	Config
	graph *crawler.Crawler
}

func (p *Pipeline) readOrCreateDoc(doc *obj.Document) (id string, err error) {
	// Query the document from the database
	databaseDoc, err := p.Ds.ReadDocBySrc(doc.Title, doc.Source)
	if err != nil {
		if !errors.Is(err, db.ErrModelNotFound) {
			// An unknown error has occured, return it
			return id, err
		} else {
			// The document does not exist.
			// Create it
			id, err = p.Ds.CreateDoc(doc)
			if err != nil {
				return id, err
			}
		}
	} else {
		// The document was found
		id = databaseDoc.Id.Hex()
	}
	return id, err
}

// Load represents the last stage of the ETL pipeline.
// This function iterates over a stream of documents and loads each document
// into the document store defined by the user.
// If the document already exists, it is not duplicated.
func (p *Pipeline) load(root *obj.Category, documents <-chan obj.Document) {
	for doc := range documents {
		docId, err := p.readOrCreateDoc(&doc)
		if err != nil {
			p.Logger.Println(err)
			continue
		}

		if err = p.Ds.AddCategory(docId, root.Id.Hex()); err != nil {
			p.Logger.Println(err)
			continue
		}
	}
}

func (p *Pipeline) Run(root *obj.Category) error {
	if p.Depth <= 0 {
		return nil
	}

	categories, documents, err := p.graph.Walk(root)
	if err != nil {
		return err
	}

	p.Depth -= 1
	go p.load(root, documents)

	for category := range categories {
		if err = p.Run(&category); err != nil {
			break
		}
	}

	return err
}
