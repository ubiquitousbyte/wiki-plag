package router

import (
	"net/http"

	"github.com/go-chi/chi/v5"
)

// A handler function that propagates errors via a return statement
// and not by writing to the response writer.
type HandlerFunc func(w http.ResponseWriter, r *http.Request) error

// Error middleware that unwraps our custom handler function
// by calling it and writing the error to the response writer, if one occured
func ErrorMiddleware(h HandlerFunc) http.HandlerFunc {
	return func(rw http.ResponseWriter, r *http.Request) {
		if err := h(rw, r); err != nil {
			WriteError(rw, err)
		}
	}
}

func URLParam(r *http.Request, key string) string {
	return chi.URLParam(r, key)
}

// A router
type Router interface {
	// The routes that this router groups together
	Routes() []Route
}

// An HTTP route
type Route interface {
	// The request handler to process requests incoming into the route
	Handler() HandlerFunc
	// The method that this route can process
	Method() string
	// The path to the route
	Path() string
}

// Create a new route that responds to GET requests on the specified path
// with the given handler function
func NewGetRoute(path string, handler HandlerFunc) Route {
	return newRoute(http.MethodGet, path, handler)
}

// Create a new route that responds to POST requests on the specified path
// with the given handler function
func NewPostRoute(path string, handler HandlerFunc) Route {
	return newRoute(http.MethodPost, path, handler)
}

// Create a new route that responds to PUT requests on the specified path
// with the given handler function
func NewPutRoute(path string, handler HandlerFunc) Route {
	return newRoute(http.MethodPut, path, handler)
}

// Create a new route that responds to DELETE requests on the specified path
// with the given handler function
func NewDeleteRoute(path string, handler HandlerFunc) Route {
	return newRoute(http.MethodDelete, path, handler)
}
