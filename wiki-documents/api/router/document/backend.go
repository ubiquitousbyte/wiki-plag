package document

import (
	"errors"
	"fmt"
	"net/http"

	"github.com/ubiquitousbyte/wiki-documents/api/router"
	"github.com/ubiquitousbyte/wiki-documents/database"
	"github.com/ubiquitousbyte/wiki-documents/entity"
)

func validateDocument(doc *entity.Document) error {
	code := http.StatusBadRequest
	if len(doc.Categories) == 0 {
		return router.NewAPIError(nil, "Document must belong to a category", code)
	}
	if len(doc.Paragraphs) == 0 {
		return router.NewAPIError(nil, "Document must contain a paragraph", code)
	}
	if len(doc.Source) == 0 {
		return router.NewAPIError(nil, "Document must have a source", code)
	}
	if len(doc.Title) == 0 {
		return router.NewAPIError(nil, "Document must have a title", code)
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
		return nil, router.NewAPIError(nil, fmt.Sprintf("Invalid document id %s", id),
			http.StatusBadRequest)
	}

	doc, err := b.store.ReadDoc(id)

	if errors.Is(err, database.ErrModelNotFound) {
		return nil, router.NewAPIError(err, fmt.Sprintf("Document %s not found", id),
			http.StatusNotFound)
	} else if err != nil {
		return nil, router.NewAPIError(nil, "Could not read document",
			http.StatusInternalServerError)
	}

	return &doc, nil
}

func (b *backend) Create(doc *entity.Document) (id entity.Id, err error) {
	if err = validateDocument(doc); err != nil {
		return id, err
	}

	doc.Id = ""
	id, err = b.store.CreateDoc(doc)
	if err != nil {
		return id, router.NewAPIError(nil, "Could not create document",
			http.StatusInternalServerError)
	}

	return
}

func (b *backend) Replace(doc *entity.Document) (err error) {
	if err = validateDocument(doc); err != nil {
		return err
	}

	err = b.store.ReplaceDoc(doc)
	if errors.Is(err, database.ErrModelNotFound) {
		return router.NewAPIError(err, fmt.Sprintf("Document %s not found", doc.Id),
			http.StatusNotFound)
	} else if err != nil {
		return router.NewAPIError(nil, "Could not update document",
			http.StatusInternalServerError)
	}

	return nil
}

func (b *backend) Delete(id entity.Id) error {
	if !id.IsValidId() {
		return router.NewAPIError(nil, fmt.Sprintf("Invalid document id %s", id),
			http.StatusBadRequest)
	}
	err := b.store.DeleteDoc(id)
	if errors.Is(err, database.ErrModelNotFound) {
		return router.NewAPIError(err, fmt.Sprintf("Document %s not found", id),
			http.StatusNotFound)
	} else if err != nil {
		return router.NewAPIError(nil, "Could not delete document",
			http.StatusInternalServerError)
	}
	return nil
}
