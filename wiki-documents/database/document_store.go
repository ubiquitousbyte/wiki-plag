package database

import "github.com/ubiquitousbyte/wiki-documents/entity"

// DocumentStore is an interface for performing CRUD operations
// on Document entities
type DocumentStore interface {
	// Reads the document by its identifier
	ReadDoc(id entity.Id) (entity.Document, error)
	// Reads the document by its title and source
	ReadDocBySrc(title, source string) (entity.Document, error)
	// Creates the document and returns its id and an error, if one occured
	CreateDoc(doc *entity.Document) (entity.Id, error)
	// Replaces the document with the entity specified as a parameter
	// The Id of the entity must be set to the id of the document
	// to be replaced
	ReplaceDoc(doc *entity.Document) error
	// Deletes the document with the given id
	DeleteDoc(id entity.Id) error
	// Adds the category with the given id to the document
	// with the given identifier
	AddCategory(docId, categoryId entity.Id) error
	// Removes the category with the specified identifier from the
	// document with the specified identifier
	RemoveCategory(docId, categoryId entity.Id) error
}
