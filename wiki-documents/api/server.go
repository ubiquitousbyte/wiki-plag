package api

import (
	"fmt"
	"net/http"

	"github.com/go-chi/chi/v5"
	"github.com/ubiquitousbyte/wiki-documents/api/router"
)

type Config struct {
	Version uint
}

type Server struct {
	cfg *Config
	mux chi.Router
}

func New(cfg *Config) *Server {
	return &Server{
		cfg: cfg,
		mux: chi.NewRouter(),
	}
}

func (s *Server) RegisterRouters(routers ...router.Router) {
	path := fmt.Sprintf("/api/v%d", s.cfg.Version)
	for _, router := range routers {
		for _, route := range router.Routes() {
			s.mux.MethodFunc(route.Method(), path+route.Path(), route.Handler())
		}
	}
}

func (s *Server) RegisterMiddleware(middleware ...func(http.Handler) http.Handler) {
	s.mux.Use(middleware...)
}

func (s *Server) ListenAndServe(address string) error {
	return http.ListenAndServe(address, s.mux)
}
