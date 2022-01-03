package router

import "net/http"

type localRoute struct {
	method  string
	path    string
	handler http.HandlerFunc
}

var _ Route = (*localRoute)(nil)

func (l *localRoute) Handler() http.HandlerFunc {
	return l.handler
}

func (l *localRoute) Method() string {
	return l.method
}

func (l *localRoute) Path() string {
	return l.path
}

func newRoute(method, path string, handler http.HandlerFunc) Route {
	route := &localRoute{method: method, path: path, handler: handler}
	return route
}
