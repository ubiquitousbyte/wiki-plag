package category

import "github.com/ubiquitousbyte/wiki-documents/api/router"

type Router struct {
	backend Backend
	routes  []router.Route
}

var _ router.Router = (*Router)(nil)

// Creates a new category router from the given backend
func NewRouter(backend Backend) *Router {
	r := &Router{backend: backend}
	r.init()
	return r
}

func (r *Router) Routes() []router.Route {
	return r.routes
}

// Initializes all category routes
func (r *Router) init() {
	r.routes = []router.Route{
		router.NewGetRoute("/categories", r.getAll),
		router.NewPostRoute("/categories", r.post),
		router.NewGetRoute("/categories/{id:.+}", r.get),
		router.NewDeleteRoute("/categories/{id:.+}", r.delete),
		router.NewGetRoute("/categories/{id:.+}/documents", r.getDocuments),
	}
}
