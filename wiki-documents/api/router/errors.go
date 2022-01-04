package router

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
)

type Error interface {
	error
	StatusCode() int
}

// Writes the error to the underlying response writer
func WriteError(w http.ResponseWriter, err error) {
	type jsonErr struct {
		Detail string `json:"detail"`
	}

	var routeErr Error
	var message string
	var status int
	if errors.As(err, &routeErr) {
		message = routeErr.Error()
		status = routeErr.StatusCode()
	} else {
		message = "Internal error"
		status = http.StatusInternalServerError
	}

	jErr := jsonErr{Detail: message}
	b, err := json.Marshal(&jErr)
	if err != nil {
		http.Error(w, "Internal error", http.StatusInternalServerError)
	} else {
		http.Error(w, string(b), status)
	}
}

type eCode int

const (
	eCodeBadId = iota + 1
	eCodeBadEntity
	eCodeNoEntity
)

func (e eCode) String() string {
	switch e {
	case eCodeBadId:
		return "Invalid id"
	case eCodeBadEntity:
		return "Invalid entity"
	case eCodeNoEntity:
		return "Entity not found"
	default:
		return "Internal error"
	}
}

var (
	ErrEntityBadId    = &entityErr{code: eCodeBadId}
	ErrEntityBad      = &entityErr{code: eCodeBadEntity}
	ErrEntityNotFound = &entityErr{code: eCodeNoEntity}
)

type entityErr struct {
	code   eCode
	detail string
	err    error
}

var _ Error = (*entityErr)(nil)

func (e entityErr) Error() string {
	var internal string
	if e.err != nil {
		internal = e.err.Error()
	}

	return fmt.Sprintf("%s (%s:%s)", e.code, e.detail, internal)
}

func (e entityErr) Unwrap() error {
	return e.err
}

func (e entityErr) Is(other error) bool {
	o, ok := other.(*entityErr)
	if !ok {
		return false
	}

	return o.code == e.code
}

func (e entityErr) StatusCode() int {
	switch e.code {
	case eCodeBadId, eCodeBadEntity:
		return http.StatusBadRequest
	case eCodeNoEntity:
		return http.StatusNotFound
	default:
		return http.StatusInternalServerError
	}
}

func (e entityErr) With(detail string) entityErr {
	el := e
	el.detail = detail
	return el
}

func (e entityErr) From(err error) error {
	el := e
	el.err = err
	return el
}
