// Package etl provides an Extract-Transform-Load interface
// for extracting documents from MediaWiki, transforming them into
// application-specific entities (Category and Document), and loading
// them into a data repository.
package etl

import (
	"github.com/ubiquitousbyte/wiki-documents/crawler"
	"github.com/ubiquitousbyte/wiki-documents/database"
	"github.com/ubiquitousbyte/wiki-documents/entity"
)

type Config struct {
	crawlCfg crawler.Config
	Ds       database.DocumentStore
	Cs       database.CategoryStore
}

type Pipeline struct {
	Config
	graph *crawler.Crawler
}

func NewPipeline(cfg Config) *Pipeline {
	return &Pipeline{
		Config: cfg,
		graph:  crawler.NewCrawler(cfg.crawlCfg),
	}
}

// readOrCreateDoc queries the document from the document storage.
// If it does not find it, the document is created
// This function returns the document's entity id or an error, if one occured
func (p *Pipeline) readOrCreateDoc(doc *entity.Document) (id entity.Id, err error) {
	databaseDoc, err := p.Ds.ReadDocBySrc(doc.Title, doc.Source)
	switch err {
	case database.ErrModelNotFound:
		return p.Ds.CreateDoc(doc)
	case nil:
		return databaseDoc.Id, nil
	default:
		return id, err
	}
}

// readOrCreateCategory queries the category from the category storage.
// If it does not find it, the category is created
// This function returns the category's entity id or an error, if one occured
func (p *Pipeline) readOrCreateCategory(c *entity.Category) (id entity.Id, err error) {
	dbCategory, err := p.Cs.ReadCategoryBySrc(c.Name, c.Source)
	switch err {
	case database.ErrModelNotFound:
		return p.Cs.CreateCategory(c)
	case nil:
		return dbCategory.Id, nil
	default:
		return id, err
	}
}

// load represents the last stage of the ETL pipeline.
// The root category and all of its children are persisted to the database
func (p *Pipeline) load(root *entity.Category, documents <-chan entity.Document) {
	createRel := true

	id, err := p.readOrCreateCategory(root)
	if err != nil {
		p.crawlCfg.Logger.Println(err)
		createRel = false
	}

	for doc := range documents {
		docId, err := p.readOrCreateDoc(&doc)
		if err != nil {
			p.crawlCfg.Logger.Println(err)
			continue
		}

		if createRel {
			if err = p.Ds.AddCategory(docId, id); err != nil {
				p.crawlCfg.Logger.Println(err)
				continue
			}
		}
	}
}

// walk traverses a layer in the graph, implicitly persisting each node
// to the respective storage.
func (p *Pipeline) walk(category *entity.Category) (<-chan entity.Category, error) {
	categories, documents, err := p.graph.Walk(category)
	if err != nil {
		return nil, err
	}
	go p.load(category, documents)
	return categories, nil
}

// BFS traverses the graph using the Breadth-First-Search algorithm and
// persists all nodes to the respective storage
func (p *Pipeline) BFS(root *entity.Category) error {
	queue := []entity.Category{*root}
	for len(queue) > 0 {
		current := queue[0]
		queue := queue[1:]

		children, err := p.walk(&current)
		if err != nil {
			return err
		}
		for child := range children {
			queue = append(queue, child)
		}
	}
	return nil
}
