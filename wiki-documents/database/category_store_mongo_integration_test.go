package database

import (
	"errors"
	"reflect"
	"testing"

	"github.com/ubiquitousbyte/wiki-documents/entity"
)

func TestReadCategories(t *testing.T) {
	tests := []struct {
		name string
		seed []entity.Category
		err  error
	}{
		{
			name: "read categories successfully",
			seed: []entity.Category{
				{
					Id:          entity.NewEntityId(),
					Name:        "Category 1",
					Description: "Category 1",
					Source:      "test",
				},
				{
					Id:          entity.NewEntityId(),
					Name:        "Category 2",
					Description: "Category 2",
					Source:      "test",
				},
				{
					Id:          entity.NewEntityId(),
					Name:        "Category 3",
					Description: "Category 3",
					Source:      "test",
				},
			},
		},
	}

	cs := mongoTestDb.categoryStore()
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			mongoTestDb.reset()
			if len(test.seed) > 0 {
				mongoTestDb.seedCategories(test.seed)
			}
			categories, err := cs.ReadCategories()
			if !errors.Is(err, test.err) {
				t.Errorf("Expected error %s, but got %s", test.err, err)
			}

			if !reflect.DeepEqual(categories, test.seed) {
				t.Errorf("Category mismatch")
			}
		})
	}
}

func TestCreateCategory(t *testing.T) {
	tests := []struct {
		name string
		c    entity.Category
		err  error
	}{
		{
			name: "create document successfully",
			c: entity.Category{
				Id:          entity.NewEntityId(),
				Name:        "Category 1",
				Description: "some text",
			},
		},
	}

	cs := mongoTestDb.categoryStore()
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			mongoTestDb.reset()
			id, err := cs.CreateCategory(&test.c)
			if !errors.Is(err, test.err) {
				t.Errorf("Expected error %s but got %s", test.err, err)
			}

			if realId := test.c.Id; realId != test.c.Id {
				t.Errorf("Expected created category id %s, but got %s", realId, id)
			}
		})
	}
}

func TestReadCategoryBySrc(t *testing.T) {
	tests := []struct {
		name   string
		source string
		title  string
		seed   []entity.Category
		err    error
	}{
		{
			name:   "read category by source returns categories",
			source: "test",
			title:  "Category 2",
			seed: []entity.Category{
				{
					Id:          entity.NewEntityId(),
					Name:        "Category 1",
					Source:      "test",
					Description: "some text",
				},
				{
					Id:          entity.NewEntityId(),
					Name:        "Category 2",
					Source:      "test",
					Description: "some text",
				},
			},
		},
	}

	cs := mongoTestDb.categoryStore()
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			mongoTestDb.reset()
			if len(test.seed) > 0 {
				mongoTestDb.seedCategories(test.seed)
			}

			c, err := cs.ReadCategoryBySrc(test.title, test.source)
			if !errors.Is(err, test.err) {
				t.Errorf("Expected error %s but got %s", test.err, err)
			}

			var match int
			for _, category := range test.seed {
				if category == c {
					match += 1
					break
				}
			}

			if match != 1 {
				t.Errorf("Category mismatch")
			}
		})
	}
}

func TestDeleteCategory(t *testing.T) {
	id := entity.NewEntityId()
	tests := []struct {
		name string
		seed []entity.Category
		id   entity.Id
		err  error
	}{
		{
			name: "delete fails when no category exists",
			err:  ErrModelNotFound,
			id:   entity.NewEntityId(),
		},
		{
			name: "delete successfully deletes category",
			seed: []entity.Category{
				{
					Id:          id,
					Name:        "Category 1",
					Source:      "test",
					Description: "some text",
				},
				{
					Id:          entity.NewEntityId(),
					Name:        "Category 2",
					Source:      "test",
					Description: "some text",
				},
			},
			id: id,
		},
	}

	cs := mongoTestDb.categoryStore()
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			mongoTestDb.reset()
			if len(test.seed) > 0 {
				mongoTestDb.seedCategories(test.seed)
			}

			err := cs.DeleteCategory(test.id)
			if !errors.Is(err, test.err) {
				t.Errorf("Expected error %s but got %s", test.err, err)
			}
		})
	}
}
