package category

import (
	"errors"
	"fmt"
	"net/http"

	"github.com/ubiquitousbyte/wiki-documents/api/router"
	"github.com/ubiquitousbyte/wiki-documents/database"
	"github.com/ubiquitousbyte/wiki-documents/entity"
)

// Validates the category and returns nil if it is valid
// or an error otherwise
func validateCategory(c *entity.Category) error {
	code := http.StatusBadRequest
	if len(c.Source) == 0 {
		return router.NewAPIError(nil, "Category source cannot be empty", code)
	}
	if len(c.Name) == 0 {
		return router.NewAPIError(nil, "Category name cannot be empty", code)
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
		return nil, router.NewAPIError(nil, fmt.Sprintf("Category id %s not valid", id),
			http.StatusBadRequest)
	}

	c, err := b.store.ReadCategory(id)
	if errors.Is(err, database.ErrModelNotFound) {
		return nil, router.NewAPIError(err, fmt.Sprintf("Category %s not found", id),
			http.StatusNotFound)
	} else if err != nil {
		return nil, router.NewAPIError(nil, "Could not read category",
			http.StatusInternalServerError)
	}

	return &c, nil
}

func (b *backend) GetDocuments(id entity.Id) ([]entity.Document, error) {
	if !id.IsValidId() {
		return nil, router.NewAPIError(nil, fmt.Sprintf("Category id %s not valid", id),
			http.StatusBadRequest)
	}

	docs, err := b.store.ReadDocsByCategory(id)
	if err != nil {
		return nil, router.NewAPIError(nil, "Cannot read documents",
			http.StatusInternalServerError)
	}

	return docs, nil
}

func (b *backend) Create(category *entity.Category) (id entity.Id, err error) {
	if err = validateCategory(category); err != nil {
		return id, err
	}

	category.Id = ""
	id, err = b.store.CreateCategory(category)
	if err != nil {
		return id, router.NewAPIError(nil, "Could not create category",
			http.StatusInternalServerError)
	}

	return
}

func (b *backend) Delete(id entity.Id) error {
	if !id.IsValidId() {
		return router.NewAPIError(nil, fmt.Sprintf("Invalid category id %s", id),
			http.StatusBadRequest)
	}

	err := b.store.DeleteCategory(id)
	if errors.Is(err, database.ErrModelNotFound) {
		return router.NewAPIError(err, fmt.Sprintf("Category %s not found", id),
			http.StatusNotFound)
	} else if err != nil {
		return router.NewAPIError(nil, "Could not read category",
			http.StatusInternalServerError)
	}

	return nil
}
