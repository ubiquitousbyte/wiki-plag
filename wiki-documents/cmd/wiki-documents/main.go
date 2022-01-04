package main

import (
	"context"
	"log"
	"os"

	"github.com/go-chi/chi/v5/middleware"
	"github.com/ubiquitousbyte/wiki-documents/api"
	"github.com/ubiquitousbyte/wiki-documents/api/router/category"
	"github.com/ubiquitousbyte/wiki-documents/api/router/document"
	"github.com/ubiquitousbyte/wiki-documents/database"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

func mongoClient(uri string) *mongo.Client {
	password, err := os.ReadFile("/run/secrets/db-password")
	if err != nil {
		panic(err)
	}
	creds := options.Credential{
		AuthSource: "wikiplag",
		Username:   "wikiplag",
		Password:   string(password),
	}
	options := options.Client().ApplyURI(uri).SetAuth(creds)
	client, err := mongo.Connect(context.Background(), options)
	if err != nil {
		panic(err)
	}
	return client
}

func main() {
	cfg := &api.Config{
		Version: 1,
	}

	server := api.New(cfg)
	server.RegisterMiddleware(
		middleware.Logger,
		middleware.AllowContentType("application/json"),
		middleware.SetHeader("Content-Type", "application/json"),
		middleware.ContentCharset("UTF-8"),
		middleware.StripSlashes,
	)

	client := mongoClient("mongodb://database:27017")

	documentStore := database.NewMongoDocumentStore(client)
	documentBackend := document.NewBackend(documentStore)
	documentRouter := document.NewRouter(documentBackend)

	categoryStore := database.NewMongoCategoryStore(client)
	categoryBackend := category.NewBackend(categoryStore)
	categoryRouter := category.NewRouter(categoryBackend)

	server.RegisterRouters(documentRouter, categoryRouter)
	log.Fatal(server.ListenAndServe(":80"))
}
