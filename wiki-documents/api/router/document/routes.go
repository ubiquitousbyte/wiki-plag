package document

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"

	"github.com/ubiquitousbyte/wiki-documents/api/router"
	"github.com/ubiquitousbyte/wiki-documents/entity"
)

func (d *Router) get(w http.ResponseWriter, r *http.Request) {
	id := entity.Id(router.URLParam(r, "id"))

	doc, err := d.backend.Get(id)
	if err != nil {
		router.WriteAPIError(w, err)
		return
	}

	if err = json.NewEncoder(w).Encode(doc); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		return
	}
}

func (d *Router) post(w http.ResponseWriter, r *http.Request) {
	var payload entity.Document
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		router.WriteAPIError(w, router.NewAPIError(nil, "Invalid document payload",
			http.StatusBadRequest))
		return
	}

	id, err := d.backend.Create(&payload)
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

func (d *Router) put(w http.ResponseWriter, r *http.Request) {
	var payload entity.Document
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		router.WriteAPIError(w, router.NewAPIError(nil, "Invalid document payload",
			http.StatusBadRequest))
		return
	}

	err := d.backend.Replace(&payload)
	if err != nil {
		apiErr, ok := err.(*router.APIError)
		if ok && apiErr.Status == http.StatusNotFound {
			d.post(w, r)
			return
		}
		router.WriteAPIError(w, err)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

func (d *Router) delete(w http.ResponseWriter, r *http.Request) {
	id := entity.Id(router.URLParam(r, "id"))

	err := d.backend.Delete(id)
	if err != nil {
		router.WriteAPIError(w, err)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}
