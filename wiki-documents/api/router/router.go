package router

import (
	"net/http"

	"github.com/go-chi/chi/v5"
)

// An HTTP route
type Route interface {
	// The request handler to process requests incoming into the route
	Handler() http.HandlerFunc
	// The method that this route can process
	Method() string
	// The path to the route
	Path() string
}

// Create a new route that responds to GET requests on the specified path
// with the given handler function
func NewGetRoute(path string, handler http.HandlerFunc) Route {
	return newRoute(http.MethodGet, path, handler)
}

// Create a new route that responds to POST requests on the specified path
// with the given handler function
func NewPostRoute(path string, handler http.HandlerFunc) Route {
	return newRoute(http.MethodPost, path, handler)
}

// Create a new route that responds to PUT requests on the specified path
// with the given handler function
func NewPutRoute(path string, handler http.HandlerFunc) Route {
	return newRoute(http.MethodPut, path, handler)
}

// Create a new route that responds to DELETE requests on the specified path
// with the given handler function
func NewDeleteRoute(path string, handler http.HandlerFunc) Route {
	return newRoute(http.MethodDelete, path, handler)
}

// A router
type Router interface {
	// The routes that this router groups together
	Routes() []Route
}

// Utility function that returns the URL parameter with the given key
// from the request
func URLParam(req *http.Request, key string) string {
	return chi.URLParam(req, key)
}
