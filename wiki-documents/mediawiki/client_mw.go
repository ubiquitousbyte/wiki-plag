package mediawiki

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"
)

// MediaWiki uses query parameters for all requests..
type mediaWikiParams url.Values

func (m mediaWikiParams) set(key string, value string) {
	url.Values(m).Set(key, value)
}

func (m mediaWikiParams) encode() string {
	return url.Values(m).Encode()
}

func newReadPagesParams(req *PageRequest) mediaWikiParams {
	params := mediaWikiParams(url.Values{
		"action":        {"query"},
		"generator":     {"categorymembers"},
		"gcmlimit":      {"1"},
		"prop":          {"extracts"},
		"explaintext":   {"true"},
		"format":        {"json"},
		"formatversion": {"2"},
		"gcmtype":       {"page|subcat"},
	})
	params.set("gcmtitle", fmt.Sprintf("Category:%s", req.category))
	if len(req.pagination) > 0 {
		params.set("gcmcontinue", req.pagination)
	}
	return params
}

func newReadCategoryParams(req *PageRequest) mediaWikiParams {
	params := mediaWikiParams(url.Values{
		"action":        {"query"},
		"prop":          {"categoryinfo|extracts"},
		"format":        {"json"},
		"formatversion": {"2"},
		"explaintext":   {"true"},
	})
	params.set("titles", fmt.Sprintf("Category:%s", req.category))
	return params
}

// MediaWiki HTTP client
type client struct {
	c *http.Client
}

// Creates a new MediaWiki HTTP client
func newHTTPClient() client {
	return client{c: &http.Client{Timeout: 10 * time.Second}}
}

func (c *client) ReadPages(request *PageRequest) ([]Page, error) {
	request.url.RawQuery = newReadPagesParams(request).encode()

	resp, err := c.c.Get(request.url.String())
	if err != nil {
		return nil, errProto.from(err)
	}
	defer resp.Body.Close()

	decoder := json.NewDecoder(resp.Body)
	var payload struct {
		Pagination struct {
			Offset   string `json:"gcmcontinue"`
			Continue string `json:"continue"`
		} `json:"continue"`
		Query struct {
			Pages []Page `json:"pages"`
		} `json:"query"`
	}
	if err = decoder.Decode(&payload); err != nil {
		return nil, errDecode.from(err)
	}

	if len(payload.Query.Pages) == 0 {
		err = fmt.Errorf("%s is empty", request.category)
		return nil, errNoPages.from(err)
	}

	if len(payload.Pagination.Offset) == 0 {
		err = io.EOF
	} else {
		request.pagination = payload.Pagination.Offset
	}

	return payload.Query.Pages, err
}

func (c *client) ReadCategory(request *PageRequest) (*Page, error) {
	request.url.RawQuery = newReadCategoryParams(request).encode()
	resp, err := c.c.Get(request.url.String())
	if err != nil {
		return nil, errProto.from(err)
	}
	defer resp.Body.Close()

	decoder := json.NewDecoder(resp.Body)
	var payload struct {
		Query struct {
			Pages []struct {
				Page
				Missing bool
			}
		}
	}
	if err = decoder.Decode(&payload); err != nil {
		return nil, errDecode.from(err)
	}

	for _, page := range payload.Query.Pages {
		if !page.Missing {
			return &page.Page, nil
		}
	}

	err = fmt.Errorf("Category %s not found", request.category)
	return nil, errNoPages.from(err)
}
