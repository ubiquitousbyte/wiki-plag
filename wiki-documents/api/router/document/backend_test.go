package document

import (
	"errors"
	"reflect"
	"testing"

	"github.com/ubiquitousbyte/wiki-documents/api/router"
	"github.com/ubiquitousbyte/wiki-documents/database"
	"github.com/ubiquitousbyte/wiki-documents/entity"
)

func TestGet(t *testing.T) {
	tests := []struct {
		name  string
		id    entity.Id
		store database.DocumentStore
		err   error
		doc   *entity.Document
	}{
		{
			name:  "invalid id returns error",
			id:    "1",
			store: &storeMock{make([]entity.Document, 0)},
			err:   router.ErrEntityBadId,
		},
		{
			name:  "get non-existent document returns error",
			id:    entity.NewEntityId(),
			store: &storeMock{make([]entity.Document, 0)},
			err:   router.ErrEntityNotFound,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			backend := NewBackend(test.store)
			doc, err := backend.Get(test.id)
			if !errors.Is(err, test.err) {
				t.Errorf("Expected error %s, but got %s", test.err, err)
			}

			if !reflect.DeepEqual(doc, test.doc) {
				t.Errorf("Document mismatch")
			}
		})
	}
}

func TestCreateInvalid(t *testing.T) {
	tests := []struct {
		name  string
		store database.DocumentStore
		doc   *entity.Document
		err   error
	}{
		{
			name:  "create document invalid categories",
			store: &storeMock{make([]entity.Document, 0)},
			doc: &entity.Document{
				Title:  "Document 1",
				Source: "test",
				Paragraphs: []entity.Paragraph{
					{
						Title:    "Paragraph 1",
						Position: 1,
						Text:     "Some text",
					},
				},
			},
			err: router.ErrEntityBad,
		},
		{
			name:  "create document invalid paragraphs",
			store: &storeMock{make([]entity.Document, 0)},
			doc: &entity.Document{
				Title:  "Document 1",
				Source: "test",
				Categories: []entity.Id{
					entity.NewEntityId(),
				},
			},
			err: router.ErrEntityBad,
		},
		{
			name:  "create document invalid title",
			store: &storeMock{make([]entity.Document, 0)},
			doc: &entity.Document{
				Title:  "",
				Source: "test",
				Categories: []entity.Id{
					entity.NewEntityId(),
				},
				Paragraphs: []entity.Paragraph{
					{
						Title:    "Paragraph 1",
						Position: 1,
						Text:     "Some text",
					},
				},
			},
			err: router.ErrEntityBad,
		},
		{
			name:  "create document invalid source",
			store: &storeMock{make([]entity.Document, 0)},
			doc: &entity.Document{
				Title: "asd",
				Categories: []entity.Id{
					entity.NewEntityId(),
				},
				Paragraphs: []entity.Paragraph{
					{
						Title:    "Paragraph 1",
						Position: 1,
						Text:     "Some text",
					},
				},
			},
			err: router.ErrEntityBad,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			backend := NewBackend(test.store)
			_, err := backend.Create(test.doc)
			if !errors.Is(err, test.err) {
				t.Errorf("Expected error %s, but got %s", test.err, err)
			}
		})
	}
}

func TestCreateSuccess(t *testing.T) {
	doc := &entity.Document{
		Title:  "Document 1",
		Source: "test",
		Categories: []entity.Id{
			entity.NewEntityId(),
		},
		Paragraphs: []entity.Paragraph{
			{
				Title:    "Paragraph 1",
				Position: 1,
				Text:     "Some text",
			},
		},
	}
	store := &storeMock{make([]entity.Document, 0)}
	backend := NewBackend(store)
	id, err := backend.Create(doc)
	if err != nil {
		t.Errorf("Unexpected error when creating document %s", err)
	}

	if !id.IsValidId() {
		t.Errorf("Invalid id returned when creating document %s", id.String())
	}
}

func TestReplaceNotFound(t *testing.T) {
	tests := []struct {
		name  string
		store database.DocumentStore
		doc   *entity.Document
		err   error
	}{
		{
			name:  "document not found",
			store: &storeMock{make([]entity.Document, 0)},
			doc: &entity.Document{
				Title:  "asd",
				Source: "test",
				Categories: []entity.Id{
					entity.NewEntityId(),
				},
				Paragraphs: []entity.Paragraph{
					{
						Title:    "Paragraph 1",
						Position: 1,
						Text:     "Some text",
					},
				},
			},
			err: router.ErrEntityNotFound,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			backend := NewBackend(test.store)
			err := backend.Replace(test.doc)
			if !errors.Is(err, test.err) {
				t.Errorf("Expected error %s, but got %s", test.err, err)
			}
		})
	}
}

func TestReplaceSuccess(t *testing.T) {
	doc := &entity.Document{
		Id:     entity.NewEntityId(),
		Title:  "asd",
		Source: "test",
		Categories: []entity.Id{
			entity.NewEntityId(),
		},
		Paragraphs: []entity.Paragraph{
			{
				Title:    "Paragraph 1",
				Position: 1,
				Text:     "Some text",
			},
		},
	}
	store := &storeMock{[]entity.Document{*doc}}

	replaced := *doc
	replaced.Title = "replaced title"

	backend := NewBackend(store)
	err := backend.Replace(&replaced)
	if err != nil {
		t.Errorf("Unexpected error when replacing document %s", err)
	}

	if store.docs[0].Title != replaced.Title {
		t.Errorf("Document not replaced")
	}
}

func TestDeleteSuccess(t *testing.T) {
	doc := &entity.Document{
		Id:     entity.NewEntityId(),
		Title:  "asd",
		Source: "test",
		Categories: []entity.Id{
			entity.NewEntityId(),
		},
		Paragraphs: []entity.Paragraph{
			{
				Title:    "Paragraph 1",
				Position: 1,
				Text:     "Some text",
			},
		},
	}
	store := &storeMock{[]entity.Document{*doc}}

	backend := NewBackend(store)
	err := backend.Delete(doc.Id)

	if err != nil {
		t.Errorf("Unexpected error when replacing document %s", err)
	}

	if len(store.docs) != 0 {
		t.Errorf("Document not deleted")
	}
}

func TestDeleteNotFound(t *testing.T) {
	tests := []struct {
		name  string
		store database.DocumentStore
		doc   *entity.Document
		err   error
	}{
		{
			name:  "document not found",
			store: &storeMock{make([]entity.Document, 0)},
			doc: &entity.Document{
				Id:     entity.NewEntityId(),
				Title:  "asd",
				Source: "test",
				Categories: []entity.Id{
					entity.NewEntityId(),
				},
				Paragraphs: []entity.Paragraph{
					{
						Title:    "Paragraph 1",
						Position: 1,
						Text:     "Some text",
					},
				},
			},
			err: router.ErrEntityNotFound,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			backend := NewBackend(test.store)
			err := backend.Delete(test.doc.Id)
			if !errors.Is(err, test.err) {
				t.Errorf("Expected error %s, but got %s", test.err, err)
			}
		})
	}
}
