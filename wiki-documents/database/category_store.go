package database

import "github.com/ubiquitousbyte/wiki-documents/entity"

// CategoryStore is an interface for performing CRUD operations
// on Category entities
type CategoryStore interface {
	// Reads all categories from the database
	ReadCategories() ([]entity.Category, error)
	// Reads all documents that are a part of this category
	ReadDocsByCategory(categoryId entity.Id) ([]entity.Document, error)
	// Reads the category with the given id
	ReadCategory(id entity.Id) (entity.Category, error)
	// Reads the category by its source and name fields
	ReadCategoryBySrc(name, source string) (entity.Category, error)
	// Creates the category
	CreateCategory(c *entity.Category) (entity.Id, error)
	// Delete category
	DeleteCategory(id entity.Id) error
}
