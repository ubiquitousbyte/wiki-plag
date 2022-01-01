package database

import (
	"errors"
	"os"
	"reflect"
	"testing"

	"github.com/ubiquitousbyte/wiki-documents/entity"
)

var mongoTestDb *mongoTestClient

func TestReadDocsByCategory(t *testing.T) {
	category := entity.NewEntityId()
	tests := []struct {
		name string
		seed []entity.Document
		err  error
	}{
		{
			name: "enmpty slice when documents not found",
			err:  nil,
			seed: nil,
		},
		{
			name: "returns the set of documents that are a part of the category",
			seed: []entity.Document{
				{
					Id:         entity.NewEntityId(),
					Title:      "Document 1",
					Categories: []entity.Id{category, entity.NewEntityId()},
					Paragraphs: []entity.Paragraph{
						{
							Title:    "Paragraph 1",
							Position: 1,
							Text:     "Some text",
						},
					},
				},
				{
					Id:         entity.NewEntityId(),
					Title:      "Document 2",
					Categories: []entity.Id{category, entity.NewEntityId()},
					Paragraphs: []entity.Paragraph{
						{
							Title:    "Paragraph 2",
							Position: 1,
							Text:     "Some text",
						},
					},
				},
			},
		},
	}

	ds := mongoTestDb.documentStore()
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			mongoTestDb.reset()
			if len(test.seed) > 0 {
				mongoTestDb.seedDocuments(test.seed)
			}
			documents, err := ds.ReadDocsByCategory(category)
			if !errors.Is(err, test.err) {
				t.Errorf("Expected error %s but got %s", test.err, err)
			}

			if !reflect.DeepEqual(documents, test.seed) {
				t.Errorf("Document mismatch")
			}
		})
	}
}

func TestCreateDoc(t *testing.T) {
	tests := []struct {
		name     string
		document entity.Document
		err      error
	}{
		{
			name: "create document successfully",
			document: entity.Document{
				Id:    entity.NewEntityId(),
				Title: "Document 1",
				Paragraphs: []entity.Paragraph{
					{
						Title:    "Paragraph 2",
						Position: 1,
						Text:     "Some text",
					},
				},
			},
		},
	}

	ds := mongoTestDb.documentStore()
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			mongoTestDb.reset()
			id, err := ds.CreateDoc(&test.document)
			if !errors.Is(err, test.err) {
				t.Errorf("Expected error %s but got %s", test.err, err)
			}

			if realId := test.document.Id; realId != test.document.Id {
				t.Errorf("Expected created document id %s, but got %s", realId, id)
			}
		})
	}
}

func TestReadDoc(t *testing.T) {
	id := entity.NewEntityId()

	tests := []struct {
		name     string
		seed     []entity.Document
		document entity.Document
		err      error
	}{
		{
			name: "read document returns successfully",
			seed: []entity.Document{
				{
					Id:    id,
					Title: "Document 1",
					Paragraphs: []entity.Paragraph{
						{
							Title:    "Paragraph 2",
							Position: 1,
							Text:     "Some text",
						},
					},
				},
			},
			document: entity.Document{
				Id:    id,
				Title: "Document 1",
				Paragraphs: []entity.Paragraph{
					{
						Title:    "Paragraph 2",
						Position: 1,
						Text:     "Some text",
					},
				},
			},
		},
	}

	ds := mongoTestDb.documentStore()
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			mongoTestDb.reset()
			if len(test.seed) > 0 {
				mongoTestDb.seedDocuments(test.seed)
			}

			doc, err := ds.ReadDoc(test.document.Id)
			if !errors.Is(err, test.err) {
				t.Errorf("Expected error %s but got %s", test.err, err)
			}

			if !reflect.DeepEqual(doc, test.document) {
				t.Errorf("Document mismatch")
			}
		})
	}
}

func TestDeleteDoc(t *testing.T) {
	id := entity.NewEntityId()
	tests := []struct {
		name string
		seed []entity.Document
		id   entity.Id
		err  error
	}{
		{
			name: "delete fails when no document exists",
			err:  ErrModelNotFound,
			id:   entity.NewEntityId(),
		},
		{
			name: "delete successfully deletes document",
			seed: []entity.Document{
				{
					Id:    id,
					Title: "Document 1",
					Paragraphs: []entity.Paragraph{
						{
							Title:    "Paragraph 2",
							Position: 1,
							Text:     "Some text",
						},
					},
				},
			},
			id: id,
		},
	}

	ds := mongoTestDb.documentStore()
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			mongoTestDb.reset()
			if len(test.seed) > 0 {
				mongoTestDb.seedDocuments(test.seed)
			}

			err := ds.DeleteDoc(test.id)
			if !errors.Is(err, test.err) {
				t.Errorf("Expected error %s but got %s", test.err, err)
			}
		})
	}
}

func TestAddCategory(t *testing.T) {
	mongoTestDb.reset()

	ds := mongoTestDb.documentStore()
	doc := entity.Document{
		Id:    entity.NewEntityId(),
		Title: "Document 1",
		Paragraphs: []entity.Paragraph{
			{
				Title:    "Paragraph 2",
				Position: 1,
				Text:     "Some text",
			},
		},
		Categories: make([]entity.Id, 0),
	}
	mongoTestDb.seedDocuments([]entity.Document{doc})

	err := ds.AddCategory(doc.Id, entity.NewEntityId())
	if err != nil {
		t.Errorf("Unexpected error when adding category %s", err)
	}

	d, err := ds.ReadDoc(doc.Id)
	if err != nil {
		t.Errorf("Unexpected error when reading doc %s", doc.Id)
	}

	if len(d.Categories) != 1 {
		t.Errorf("Expected category count of doc to be 1 after adding category")
	}
}

func TestRemoveCategory(t *testing.T) {
	mongoTestDb.reset()

	ds := mongoTestDb.documentStore()
	doc := entity.Document{
		Id:    entity.NewEntityId(),
		Title: "Document 1",
		Paragraphs: []entity.Paragraph{
			{
				Title:    "Paragraph 2",
				Position: 1,
				Text:     "Some text",
			},
		},
		Categories: []entity.Id{entity.NewEntityId()},
	}
	mongoTestDb.seedDocuments([]entity.Document{doc})

	err := ds.RemoveCategory(doc.Id, doc.Categories[0])
	if err != nil {
		t.Errorf("Unexpected error when adding category %s", err)
	}

	d, err := ds.ReadDoc(doc.Id)
	if err != nil {
		t.Errorf("Unexpected error when reading doc %s", doc.Id)
	}

	if len(d.Categories) != 0 {
		t.Errorf("Expected category count of doc to be 0 after removing category")
	}
}

func TestMain(m *testing.M) {
	mongoTestDb = newMongoTestClient()
	code := m.Run()
	mongoTestDb.close()
	os.Exit(code)
}
