package router

import (
	"net/http"

	"github.com/go-chi/chi/v5"
)

type Route interface {
	Handler() http.HandlerFunc
	Method() string
	Path() string
}

func NewGetRoute(path string, handler http.HandlerFunc) Route {
	return newRoute(http.MethodGet, path, handler)
}

func NewPostRoute(path string, handler http.HandlerFunc) Route {
	return newRoute(http.MethodPost, path, handler)
}

func NewPutRoute(path string, handler http.HandlerFunc) Route {
	return newRoute(http.MethodPut, path, handler)
}

func NewDeleteRoute(path string, handler http.HandlerFunc) Route {
	return newRoute(http.MethodDelete, path, handler)
}

type Router interface {
	Routes() []Route
}

func URLParam(req *http.Request, key string) string {
	return chi.URLParam(req, key)
}

/*type Router struct {
	router chi.Router
}

func NewRouter(documentRouter *DocumentRouter, categoryRouter *CategoryRouter) Router {
	router := Router{}
	router.router = chi.NewRouter()
	router.router.Use(
		middleware.AllowContentEncoding("application/json"),
		middleware.ContentCharset("UTF-8"),
		middleware.Logger,
		middleware.StripSlashes,
	)

	router.


	return Router{router: router.router}
}

func (r *Router) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	r.router.ServeHTTP(w, req)
}*/
