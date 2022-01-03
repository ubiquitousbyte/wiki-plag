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

func (c *Router) getAll(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query()
	start, err := strconv.ParseUint(query.Get("start"), 10, strconv.IntSize)
	if err != nil {
		router.WriteAPIError(w, router.NewAPIError(nil, "Invalid start", http.StatusBadRequest))
		return
	}

	offset, err := strconv.ParseUint(query.Get("offset"), 10, strconv.IntSize)
	if err != nil {
		router.WriteAPIError(w, router.NewAPIError(nil, "Invalid offset", http.StatusBadRequest))
		return
	}

	categories, err := c.backend.GetAll(uint(start), uint(offset))
	if err != nil {
		router.WriteAPIError(w, err)
		return
	}

	if err = json.NewEncoder(w).Encode(&categories); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		return
	}
}

func (c *Router) get(w http.ResponseWriter, r *http.Request) {
	id := entity.Id(router.URLParam(r, "id"))

	category, err := c.backend.Get(id)
	if err != nil {
		router.WriteAPIError(w, err)
		return
	}

	if err = json.NewEncoder(w).Encode(category); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		return
	}
}

func (c *Router) getDocuments(w http.ResponseWriter, r *http.Request) {
	id := entity.Id(router.URLParam(r, "id"))

	documents, err := c.backend.GetDocuments(id)
	if err != nil {
		router.WriteAPIError(w, err)
		return
	}

	if err := json.NewEncoder(w).Encode(&documents); err != nil {
		router.WriteAPIError(w, err)
		return
	}
}

func (c *Router) post(w http.ResponseWriter, r *http.Request) {
	var payload entity.Category
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		router.WriteAPIError(w, router.NewAPIError(nil, "Invalid category in payload",
			http.StatusBadRequest))
		return
	}

	id, err := c.backend.Create(&payload)
	if err != nil {
		router.WriteAPIError(w, err)
		return
	}

	location, err := url.Parse(fmt.Sprintf("%s/%s", r.URL.Path, id))
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	w.Header().Set("Location", location.String())
	w.WriteHeader(http.StatusCreated)
}

func (c *Router) delete(w http.ResponseWriter, r *http.Request) {
	id := entity.Id(router.URLParam(r, "id"))

	err := c.backend.Delete(id)
	if err != nil {
		router.WriteAPIError(w, err)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}
