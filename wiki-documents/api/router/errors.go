package router

import (
	"encoding/json"
	"net/http"
)

type APIError struct {
	Cause  error  `json:"-"`
	Detail string `json:"detail"`
	Status int    `json:"-"`
}

func (e *APIError) Error() string {
	if e.Cause == nil {
		return e.Detail
	}
	return e.Detail + " : " + e.Cause.Error()
}

func NewAPIError(cause error, detail string, status int) error {
	return &APIError{
		Cause:  cause,
		Detail: detail,
		Status: status,
	}
}

func WriteAPIError(w http.ResponseWriter, err error) {
	apiErr, ok := err.(*APIError)
	if !ok {
		apiErr = &APIError{Detail: err.Error(), Status: http.StatusInternalServerError}
	}
	b, err := json.Marshal(apiErr)
	if err != nil {
		http.Error(w, apiErr.Error(), apiErr.Status)
	} else {
		http.Error(w, string(b), apiErr.Status)
	}

}
