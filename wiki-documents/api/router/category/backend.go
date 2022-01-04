package category

import (
	"errors"
	"fmt"

	"github.com/ubiquitousbyte/wiki-documents/api/router"
	"github.com/ubiquitousbyte/wiki-documents/database"
	"github.com/ubiquitousbyte/wiki-documents/entity"
)

// Validates the category and returns nil if it is valid
// or an error otherwise
func validateCategory(c *entity.Category) error {
	if len(c.Source) == 0 {
		return router.ErrEntityBad.With("Category source cannot be empty")
	}
	if len(c.Name) == 0 {
		return router.ErrEntityBad.With("Category name cannot be empty")
	}
	return nil
}

// A category backend
type Backend interface {
	// Retrieves a batch of categories starting at index.
	// The number of categories retrieved is defined by offset.
	GetAll(index, offset uint) ([]entity.Category, error)
	// Retrieves the category with the id
	Get(id entity.Id) (*entity.Category, error)
	// Retrieves all documents that are a part of the category with the given id
	GetDocuments(id entity.Id) ([]entity.Document, error)
	// Creates a category.
	// If the category is not valid, an error is returned
	Create(category *entity.Category) (entity.Id, error)
	// Deletes the category with the specified id
	Delete(id entity.Id) error
}

type backend struct {
	store database.CategoryStore
}

// Creates a new backend from the category store
func NewBackend(store database.CategoryStore) *backend {
	return &backend{store: store}
}

func (b *backend) GetAll(start, offset uint) ([]entity.Category, error) {
	end := start + offset

	categories, err := b.store.ReadCategories()
	if err != nil {
		return make([]entity.Category, 0), nil
	}

	count := uint(len(categories))
	if count < start {
		return make([]entity.Category, 0), nil
	}

	if end > count {
		end = count
	}

	return categories[start:end], nil
}

func (b *backend) Get(id entity.Id) (*entity.Category, error) {
	if !id.IsValidId() {
		return nil, router.ErrEntityBadId.With(id.String())
	}

	c, err := b.store.ReadCategory(id)
	if errors.Is(err, database.ErrModelNotFound) {
		return nil, router.ErrEntityNotFound.With(id.String())
	} else if err != nil {
		return nil, err
	}

	return &c, nil
}

func (b *backend) GetDocuments(id entity.Id) ([]entity.Document, error) {
	if !id.IsValidId() {
		return nil, router.ErrEntityBadId.With(id.String())
	}

	docs, err := b.store.ReadDocsByCategory(id)
	if err != nil {
		return nil, err
	}

	return docs, nil
}

func (b *backend) Create(category *entity.Category) (id entity.Id, err error) {
	if err = validateCategory(category); err != nil {
		return id, err
	}

	_, err = b.store.ReadCategoryBySrc(category.Name, category.Source)
	if errors.Is(err, database.ErrModelNotFound) {
		category.Id = ""
		id, err = b.store.CreateCategory(category)
		return
	} else if err == nil {
		msg := fmt.Sprintf("Category with name %s and source %s already exists",
			category.Name, category.Source)
		return id, router.ErrEntityBad.With(msg)
	} else {
		return id, err
	}
}

func (b *backend) Delete(id entity.Id) error {
	if !id.IsValidId() {
		return router.ErrEntityBadId.With(id.String())
	}
	err := b.store.DeleteCategory(id)
	if errors.Is(err, database.ErrModelNotFound) {
		return router.ErrEntityNotFound.With(id.String())
	}
	return err
}
