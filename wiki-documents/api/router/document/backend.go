package document

import (
	"errors"
	"fmt"

	"github.com/ubiquitousbyte/wiki-documents/api/router"
	"github.com/ubiquitousbyte/wiki-documents/database"
	"github.com/ubiquitousbyte/wiki-documents/entity"
)

func validateDocument(doc *entity.Document) error {
	if len(doc.Categories) == 0 {
		return router.ErrEntityBad.With("Document must have categories")
	}
	if len(doc.Paragraphs) == 0 {
		return router.ErrEntityBad.With("Document must have paragraphs")
	}
	if len(doc.Source) == 0 {
		return router.ErrEntityBad.With("Document must have a source")
	}
	if len(doc.Title) == 0 {
		return router.ErrEntityBad.With("Document must have a title")
	}
	return nil
}

type Backend interface {
	Get(id entity.Id) (*entity.Document, error)
	Create(document *entity.Document) (entity.Id, error)
	Replace(document *entity.Document) error
	Delete(id entity.Id) error
}

type backend struct {
	store database.DocumentStore
}

func NewBackend(store database.DocumentStore) *backend {
	return &backend{store: store}
}

func (b *backend) Get(id entity.Id) (*entity.Document, error) {
	if !id.IsValidId() {
		return nil, router.ErrEntityBadId.With(id.String())
	}

	doc, err := b.store.ReadDoc(id)
	if errors.Is(err, database.ErrModelNotFound) {
		return nil, router.ErrEntityNotFound.With(id.String())
	} else if err != nil {
		return nil, err
	}

	return &doc, nil
}

func (b *backend) Create(doc *entity.Document) (id entity.Id, err error) {
	if err = validateDocument(doc); err != nil {
		return id, err
	}

	_, err = b.store.ReadDocBySrc(doc.Title, doc.Source)
	if errors.Is(err, database.ErrModelNotFound) {
		doc.Id = ""
		id, err = b.store.CreateDoc(doc)
		return
	} else if err == nil {
		msg := fmt.Sprintf("Document with name %s and source %s already exists",
			doc.Title, doc.Source)
		return id, router.ErrEntityBad.With(msg)
	} else {
		return id, err
	}
}

func (b *backend) Replace(doc *entity.Document) (err error) {
	if err = validateDocument(doc); err != nil {
		return err
	}

	err = b.store.ReplaceDoc(doc)
	if errors.Is(err, database.ErrModelNotFound) {
		return router.ErrEntityNotFound.With(doc.Id.String())
	}

	return
}

func (b *backend) Delete(id entity.Id) error {
	if !id.IsValidId() {
		return router.ErrEntityBadId.With(id.String())
	}

	err := b.store.DeleteDoc(id)
	if errors.Is(err, database.ErrModelNotFound) {
		return router.ErrEntityNotFound.With(id.String())
	}

	return err
}
