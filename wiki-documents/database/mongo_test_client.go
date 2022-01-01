package database

import (
	"context"

	"github.com/ubiquitousbyte/wiki-documents/entity"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

// A test client for MongoDB that can be used for integration tests
type mongoTestClient struct {
	c *mongo.Client
}

func newMongoTestClient() *mongoTestClient {
	auth := options.Credential{
		Username: "root",
		Password: "root",
	}
	uri := "mongodb://root:root@localhost:27020"

	opts := options.Client().ApplyURI(uri).SetAuth(auth)
	client, err := mongo.Connect(context.Background(), opts)
	if err != nil {
		panic(err)
	}

	if err = client.Ping(context.Background(), nil); err != nil {
		panic(err)
	}

	return &mongoTestClient{c: client}
}

func (m *mongoTestClient) close() {
	defer m.c.Disconnect(context.Background())

	db := m.c.Database("wikiplag")
	coll := db.Collection("documents")

	if err := coll.Drop(context.Background()); err != nil {
		panic(err)
	}
}

func (m *mongoTestClient) reset() {
	db := m.c.Database("wikiplag")
	coll := db.Collection("documents")

	if _, err := coll.DeleteMany(context.Background(), bson.D{}); err != nil {
		panic(err)
	}
}

func (m *mongoTestClient) seedDocuments(docs []entity.Document) {
	var data []interface{}
	for _, doc := range docs {
		data = append(data, doc)
	}
	db := m.c.Database("wikiplag")
	coll := db.Collection("documents")
	if _, err := coll.InsertMany(context.Background(), data); err != nil {
		panic(err)
	}
}

func (m *mongoTestClient) seedCategories(categories []entity.Category) {
	var data []interface{}
	for _, cat := range categories {
		data = append(data, cat)
	}
	db := m.c.Database("wikiplag")
	coll := db.Collection("documents")
	if _, err := coll.InsertMany(context.Background(), data); err != nil {
		panic(err)
	}
}

func (m *mongoTestClient) documentStore() *MongoDocumentStore {
	return NewMongoDocumentStore(m.c)
}

func (m *mongoTestClient) categoryStore() *MongoCategoryStore {
	return NewMongoCategoryStore(m.c)
}
