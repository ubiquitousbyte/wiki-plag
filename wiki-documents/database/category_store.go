package database

import obj "github.com/ubiquitousbyte/wiki-documents/models"

// CategoryStore is an interface for performing CRUD operations
// on Category entities
type CategoryStore interface {
	// Reads all categories from the database
	ReadCategories() ([]obj.Category, error)
	// Reads the category with the given name and source
	ReadCategory(id string) (obj.Category, error)
}
