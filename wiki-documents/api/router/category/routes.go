package category

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"

	"github.com/ubiquitousbyte/wiki-documents/api/router"
	"github.com/ubiquitousbyte/wiki-documents/entity"
)

type paginationErr struct {
	detail string
}

func (p paginationErr) Error() string {
	return p.detail
}

func (p paginationErr) StatusCode() int {
	return http.StatusBadRequest
}

func (c *Router) getAll(w http.ResponseWriter, r *http.Request) error {
	query := r.URL.Query()
	start, err := strconv.ParseUint(query.Get("start"), 10, strconv.IntSize)
	if err != nil {
		return paginationErr{detail: "Invalid start"}
	}

	offset, err := strconv.ParseUint(query.Get("offset"), 10, strconv.IntSize)
	if err != nil {
		return paginationErr{detail: "Invalid offset"}
	}

	categories, err := c.backend.GetAll(uint(start), uint(offset))
	if err != nil {
		return err
	}

	return json.NewEncoder(w).Encode(&categories)
}

func (c *Router) get(w http.ResponseWriter, r *http.Request) error {
	id := entity.Id(router.URLParam(r, "id"))

	category, err := c.backend.Get(id)
	if err != nil {
		return err
	}

	return json.NewEncoder(w).Encode(category)
}

func (c *Router) getDocuments(w http.ResponseWriter, r *http.Request) error {
	id := entity.Id(router.URLParam(r, "id"))

	documents, err := c.backend.GetDocuments(id)
	if err != nil {
		return err
	}

	return json.NewEncoder(w).Encode(&documents)
}

func (c *Router) post(w http.ResponseWriter, r *http.Request) error {
	var payload entity.Category
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		return err
	}

	id, err := c.backend.Create(&payload)
	if err != nil {
		return err
	}

	location, err := url.Parse(fmt.Sprintf("%s/%s", r.URL.Path, id))
	if err != nil {
		return err
	}

	w.Header().Set("Location", location.String())
	w.WriteHeader(http.StatusCreated)
	return nil
}

func (c *Router) delete(w http.ResponseWriter, r *http.Request) error {
	id := entity.Id(router.URLParam(r, "id"))

	err := c.backend.Delete(id)
	if err != nil {
		return err
	}

	w.WriteHeader(http.StatusNoContent)
	return nil
}
