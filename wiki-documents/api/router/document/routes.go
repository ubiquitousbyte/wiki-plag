package document

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"

	"github.com/ubiquitousbyte/wiki-documents/api/router"
	"github.com/ubiquitousbyte/wiki-documents/entity"
)

func (d *Router) get(w http.ResponseWriter, r *http.Request) error {
	id := entity.Id(router.URLParam(r, "id"))

	doc, err := d.backend.Get(id)
	if err != nil {
		return err
	}

	return json.NewEncoder(w).Encode(doc)
}

func (d *Router) post(w http.ResponseWriter, r *http.Request) error {
	var payload entity.Document
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		return router.ErrEntityBad
	}

	id, err := d.backend.Create(&payload)
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

func (d *Router) put(w http.ResponseWriter, r *http.Request) error {
	var payload entity.Document
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		return router.ErrEntityBad
	}

	err := d.backend.Replace(&payload)
	if err != nil {
		var routeErr router.Error
		if errors.As(err, &routeErr) {
			if routeErr.StatusCode() == http.StatusNotFound {
				return d.post(w, r)
			}
		}
		return err
	}

	w.WriteHeader(http.StatusNoContent)
	return nil
}

func (d *Router) delete(w http.ResponseWriter, r *http.Request) error {
	id := entity.Id(router.URLParam(r, "id"))

	err := d.backend.Delete(id)
	if err != nil {
		return err
	}

	w.WriteHeader(http.StatusNoContent)
	return nil
}
