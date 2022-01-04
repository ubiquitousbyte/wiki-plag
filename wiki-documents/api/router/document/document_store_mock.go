package document

import (
	"github.com/ubiquitousbyte/wiki-documents/database"
	"github.com/ubiquitousbyte/wiki-documents/entity"
)

type storeMock struct {
	docs []entity.Document
}

var _ database.DocumentStore = (*storeMock)(nil)

func (m *storeMock) ReadDoc(id entity.Id) (entity.Document, error) {
	for _, doc := range m.docs {
		if doc.Id == id {
			return doc, nil
		}
	}
	return entity.Document{}, database.ErrModelNotFound
}

func (m *storeMock) ReadDocBySrc(title, source string) (entity.Document, error) {
	for _, doc := range m.docs {
		if doc.Title == title && doc.Source == source {
			return doc, nil
		}
	}
	return entity.Document{}, database.ErrModelNotFound
}

func (m *storeMock) CreateDoc(doc *entity.Document) (entity.Id, error) {
	doc.Id = entity.NewEntityId()
	m.docs = append(m.docs, *doc)
	return doc.Id, nil
}

func (m *storeMock) ReplaceDoc(doc *entity.Document) error {
	for i := 0; i < len(m.docs); i++ {
		if m.docs[i].Id == doc.Id {
			m.docs[i] = *doc
			return nil
		}
	}
	return database.ErrModelNotFound
}

func (m *storeMock) DeleteDoc(id entity.Id) error {
	for i := 0; i < len(m.docs); i++ {
		if m.docs[i].Id == id {
			m.docs = make([]entity.Document, len(m.docs)-1)
			m.docs = append(m.docs, m.docs[:i]...)
			if i < len(m.docs)-1 {
				m.docs = append(m.docs, m.docs[i+1:]...)
			}
			return nil
		}
	}
	return database.ErrModelNotFound
}

func (m *storeMock) AddCategory(docId, categoryId entity.Id) error {
	return nil
}

func (m *storeMock) RemoveCategory(docId, categoryId entity.Id) error {
	return nil
}
