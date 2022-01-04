package mediawiki

import (
	"errors"
	"testing"
)

func TestNewPageRequest(t *testing.T) {
	tests := []struct {
		name     string
		lang     string
		category string
		err      error
	}{
		{
			name:     "invalid language identifier",
			lang:     "bb",
			category: "Informatik",
			err:      ErrInvalidRequest,
		},
		{
			name: "empty category",
			lang: "de",
			err:  ErrInvalidRequest,
		},
		{
			name:     "valid inputs",
			lang:     "de",
			category: "Informatik",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := NewPageRequest(test.lang, test.category)
			if !errors.Is(err, test.err) {
				t.Errorf("Expected error %s, but got %s instead", test.err, err)
			}
		})
	}
}
