package document

import "github.com/ubiquitousbyte/wiki-documents/api/router"

type Router struct {
	backend Backend
	routes  []router.Route
}

func NewRouter(backend Backend) *Router {
	d := &Router{backend: backend}
	d.init()
	return d
}

func (r *Router) Routes() []router.Route {
	return r.routes
}

func (r *Router) init() {
	r.routes = []router.Route{
		router.NewPostRoute("/document", r.post),
		router.NewGetRoute("/document/{id:.+}", r.get),
		router.NewPutRoute("/document/{id:.+}", r.put),
		router.NewDeleteRoute("/document/{id:.+}", r.delete),
	}
}
