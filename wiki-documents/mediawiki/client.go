// Package mediawiki provides an endpoint to Wikipedia's REST API, also known
// as MediaWiki.
package mediawiki

import (
	"errors"
	"fmt"
	"net/url"

	"golang.org/x/text/language"
)

// MediaWiki aggregates similar contents into namespaces.
// Every namespace is represented via an unique integer.
// See https://www.mediawiki.org/wiki/Help:Namespaces
type Namespace int

const (
	// The main namespace. This is where documents are stored
	NamespaceMain Namespace = iota
	// The category namespace. This is where categories are stored
	NamespaceCategory Namespace = 14
)

// Endpoint format for establishing a connection to MediaWiki
const epFormat = "https://%s.wikipedia.org/w/api.php?"

// A wikipedia page
type Page struct {
	Id        int       `json:"pageid"`  // The id of the page
	Namespace Namespace `json:"ns"`      // The namespace the page is a part of
	Title     string    `json:"title"`   // The title of the page
	Text      string    `json:"extract"` // Any text related to the page
}

// A wikipedia page request
type PageRequest struct {
	url        *url.URL // The endpoint to connect to
	category   string   // The category to extract information for
	pagination string   // Optional pagination used to execute multipart queries
}

// Creates a new Wikipedia page request.
// The user must provide an ISO 639-1 compliant language specifier, e.g
// "de", "en", "it", "fr".
// The identifier is used to extract the pages from the language-specific
// MediaWiki subsystem.
func NewPageRequest(lang, category string) (*PageRequest, error) {
	if len(category) == 0 {
		return nil, ErrInvalidRequest.from(errors.New("Empty category name"))
	}
	if len(lang) != 2 {
		err := fmt.Errorf("Invalid ISO 639-1 language identifier %s", lang)
		return nil, ErrInvalidRequest.from(err)
	}

	l, err := language.ParseBase(lang)
	if err != nil {
		return nil, ErrInvalidRequest.from(err)
	}

	ep, err := url.Parse(fmt.Sprintf(epFormat, l))
	if err != nil {
		return nil, ErrInvalidRequest.from(err)
	}

	return &PageRequest{url: ep, category: category}, nil
}

func (p *PageRequest) String() string {
	return fmt.Sprintf("<PageRequest>: %s %s", p.url, p.category)
}

// A MediaWiki client that is able to extract wikipedia pages.
type Client interface {
	// ReadPages reads a batch of pages that are part of the category specified
	// in the PageRequest.
	//
	// The caller can read until an io.EOF error is returned. This indicates
	// that all pages associated with the request have been returned.
	// If err is nil, then the caller can continue reading.
	//
	// Note:
	// Take a look at the MediaWiki Code of Conduct and try to emit requests
	// serially to avoid clogging the API.
	// See https://www.mediawiki.org/wiki/API:Etiquette#Request_limit
	ReadPages(request *PageRequest) (pages []Page, err error)

	// Reads a category from the MediaWiki API.
	// If the category does not exist, an errNoPage error is returned
	ReadCategory(request *PageRequest) (*Page, error)
}

// Creates a new client to MediaWiki
func NewClient() Client {
	httpClient := newHTTPClient()
	return &httpClient
}
