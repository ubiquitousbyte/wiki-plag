package router

type localRoute struct {
	method  string
	path    string
	handler HandlerFunc
}

var _ Route = (*localRoute)(nil)

func (l *localRoute) Handler() HandlerFunc {
	return l.handler
}

func (l *localRoute) Method() string {
	return l.method
}

func (l *localRoute) Path() string {
	return l.path
}

func newRoute(method, path string, handler HandlerFunc) Route {
	route := &localRoute{method: method, path: path, handler: handler}
	return route
}
