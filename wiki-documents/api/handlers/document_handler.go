// Package handlers implements HTTP request handlers
// for application-specific entities.
package handlers

import "github.com/ubiquitousbyte/wiki-documents/database"

type DocumentHandler struct {
	storage *database.DocumentStore
}
