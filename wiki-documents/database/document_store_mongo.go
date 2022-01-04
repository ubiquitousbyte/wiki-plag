package database

import (
	"context"
	"time"

	"github.com/ubiquitousbyte/wiki-documents/entity"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
)

type MongoDocumentStore struct {
	c *mongo.Client
}

var _ DocumentStore = (*MongoDocumentStore)(nil)

func NewMongoDocumentStore(client *mongo.Client) *MongoDocumentStore {
	return &MongoDocumentStore{c: client}
}

func (m *MongoDocumentStore) collection() *mongo.Collection {
	return m.c.Database("wikiplag").Collection("documents")
}

func (m *MongoDocumentStore) ReadDoc(id entity.Id) (doc entity.Document, err error) {
	coll := m.collection()
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	filter := bson.M{"_id": id, "paragraphs": bson.M{"$exists": true}}
	res := coll.FindOne(ctx, filter)

	if err = res.Err(); err != nil {
		if err == mongo.ErrNoDocuments {
			err = ErrModelNotFound.from(err)
		} else {
			err = dbErr{}.from(err)
		}
		return doc, err
	}

	if err = res.Decode(&doc); err != nil {
		return doc, ErrInvalidModel.from(err)
	}

	return
}

func (m *MongoDocumentStore) ReadDocBySrc(title, source string) (entity.Document, error) {
	var doc entity.Document

	coll := m.collection()
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	filter := bson.M{"title": title, "source": source}
	res := coll.FindOne(ctx, filter)
	if err := res.Err(); err != nil {
		if err == mongo.ErrNoDocuments {
			err = ErrModelNotFound.from(err)
		} else {
			err = dbErr{}.from(err)
		}
		return doc, err
	}

	if err := res.Decode(&doc); err != nil {
		return doc, ErrInvalidModel.from(err)
	}

	return doc, nil
}

func (m *MongoDocumentStore) CreateDoc(doc *entity.Document) (id entity.Id, err error) {
	coll := m.collection()
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	res, err := coll.InsertOne(ctx, doc)
	if err != nil {
		return id, ErrCreate.from(err)
	}

	id = entity.Id(res.InsertedID.(primitive.ObjectID).Hex())
	return
}

func (m *MongoDocumentStore) AddCategory(doc, cat entity.Id) error {
	coll := m.collection()
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	filter := bson.M{"_id": doc}
	update := bson.M{"$addToSet": bson.M{"categories": cat}}
	res := coll.FindOneAndUpdate(ctx, filter, update)
	if err := res.Err(); err != nil {
		if err == mongo.ErrNoDocuments {
			err = ErrUpdate.from(ErrModelNotFound.from(err))
		} else {
			err = ErrUpdate.from(err)
		}
		return err
	}

	return nil
}

func (m *MongoDocumentStore) RemoveCategory(doc, cat entity.Id) error {
	coll := m.collection()
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	filter := bson.M{"_id": doc}
	update := bson.M{
		"$pull": bson.M{
			"categories": bson.M{"$in": []entity.Id{cat}},
		},
	}
	res := coll.FindOneAndUpdate(ctx, filter, update)
	if err := res.Err(); err != nil {
		if err == mongo.ErrNoDocuments {
			err = ErrUpdate.from(ErrModelNotFound.from(err))
		} else {
			err = ErrUpdate.from(err)
		}
		return err
	}

	return nil
}

func (m *MongoDocumentStore) ReplaceDoc(doc *entity.Document) error {
	coll := m.collection()
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	filter := bson.M{"_id": doc.Id, "paragraphs": bson.M{"$exists": true}}
	res := coll.FindOneAndReplace(ctx, filter, doc)
	if err := res.Err(); err != nil {
		if err == mongo.ErrNoDocuments {
			err = ErrUpdate.from(ErrModelNotFound.from(err))
		} else {
			err = ErrUpdate.from(err)
		}
		return err
	}
	return nil
}

func (m *MongoDocumentStore) DeleteDoc(id entity.Id) error {
	coll := m.collection()
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	filter := bson.M{"_id": id, "paragraphs": bson.M{"$exists": true}}
	res := coll.FindOneAndDelete(ctx, filter)
	if err := res.Err(); err != nil {
		if err == mongo.ErrNoDocuments {
			err = ErrDelete.from(ErrModelNotFound.from(err))
		} else {
			err = ErrDelete.from(err)
		}
		return err
	}

	return nil
}
