package mediawiki

import (
	"errors"
	"io"
	"testing"
)

func TestReadPages(t *testing.T) {
	tests := []struct {
		name         string
		lang         string
		category     string
		minPageCount int
		err          error
	}{
		{
			name:         "read pages from a valid category",
			lang:         "de",
			category:     "Informatik",
			minPageCount: 90,
		},
	}

	client := NewClient()
	for _, test := range tests {
		count := 0
		request, _ := NewPageRequest(test.lang, test.category)
		for {
			batch, err := client.ReadPages(request)
			if err == io.EOF {
				break
			}

			if !errors.Is(err, test.err) {
				t.Errorf("Expected error %s, but got %s", test.err, err)

			}
			count += len(batch)
		}

		if count < test.minPageCount {
			t.Errorf("Expected at least %d pages, but got %d", test.minPageCount, count)
		}
	}
}

func TestReadCategory(t *testing.T) {
	tests := []struct {
		name     string
		lang     string
		category string
		err      error
	}{
		{
			name:     "category does not exist",
			lang:     "de",
			category: "Asdasdasd",
			err:      ErrNoPages,
		},
		{
			name:     "category found",
			lang:     "de",
			category: "Informatik",
			err:      nil,
		},
	}

	client := NewClient()
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request, _ := NewPageRequest(test.lang, test.category)
			_, err := client.ReadCategory(request)
			if !errors.Is(err, test.err) {
				t.Errorf("Expected error %s, but got %s", test.err, err)
			}
		})
	}
}
