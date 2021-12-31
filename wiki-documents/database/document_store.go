package database

import (
	obj "github.com/ubiquitousbyte/wiki-documents/models"
)

// DocumentStore is an interface for performing CRUD operations
// on Document entities
type DocumentStore interface {
	// Reads all documents that are a part of this category
	ReadDocsByCategory(categoryId string) ([]obj.Document, error)
	// Reads the document by its identifier
	ReadDoc(id string) (obj.Document, error)
	// Reads the document by its title and source
	ReadDocBySrc(title, source string) (obj.Document, error)
	// Creates the document
	CreateDoc(doc *obj.Document) (string, error)
	// Adds the category with the given id to the documet
	// with the given identifier
	AddCategory(docId, categoryId string) error
	// Removes the category with the specified identifier from the
	// document with the specified identifier
	RemoveCategory(docId, categoryId string) error
	// Deletes the document with the given id
	DeleteDoc(id string) error
}
