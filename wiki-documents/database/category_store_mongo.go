package database

import (
	"context"
	"time"

	"github.com/ubiquitousbyte/wiki-documents/entity"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
)

type MongoCategoryStore struct {
	c *mongo.Client
}

var _ CategoryStore = (*MongoCategoryStore)(nil)

func NewMongoCategoryStore(client *mongo.Client) *MongoCategoryStore {
	return &MongoCategoryStore{c: client}
}

func (m *MongoCategoryStore) collection() *mongo.Collection {
	return m.c.Database("wikiplag").Collection("documents")
}

func (m *MongoCategoryStore) ReadCategories() ([]entity.Category, error) {
	coll := m.collection()
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	filter := bson.M{"name": bson.M{"$exists": true}}
	cursor, err := coll.Find(ctx, filter)
	if err != nil {
		return nil, ErrModelNotFound.from(err)
	}

	var categories []entity.Category
	if err = cursor.All(ctx, &categories); err != nil {
		return nil, ErrInvalidModel.from(err)
	}

	return categories, nil
}

func (m *MongoCategoryStore) ReadCategory(id entity.Id) (c entity.Category, err error) {
	coll := m.collection()
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	filter := bson.M{"_id": id, "name": bson.M{"$exists": true}}
	res := coll.FindOne(ctx, filter)
	if err := res.Err(); err != nil {
		return c, ErrModelNotFound.from(err)
	}

	if err = res.Decode(&c); err != nil {
		return c, ErrInvalidModel.from(err)
	}

	return
}

func (m *MongoCategoryStore) ReadCategoryBySrc(name, source string) (entity.Category, error) {
	var c entity.Category

	coll := m.collection()
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	filter := bson.M{"name": name, "source": source}
	res := coll.FindOne(ctx, filter)
	if err := res.Err(); err != nil {
		if err == mongo.ErrNoDocuments {
			err = ErrModelNotFound.from(err)
		} else {
			err = dbErr{}.from(err)
		}
		return c, err
	}

	if err := res.Decode(&c); err != nil {
		return c, ErrInvalidModel.from(err)
	}

	return c, nil
}

func (m *MongoCategoryStore) CreateCategory(c *entity.Category) (id entity.Id, err error) {
	coll := m.collection()
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	res, err := coll.InsertOne(ctx, c)
	if err != nil {
		return id, ErrCreate.from(err)
	}

	id = entity.Id(res.InsertedID.(primitive.ObjectID).Hex())
	return
}

func (m *MongoCategoryStore) DeleteCategory(id entity.Id) error {
	coll := m.collection()
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	filter := bson.M{"_id": id, "name": bson.M{"$exists": true}}
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
