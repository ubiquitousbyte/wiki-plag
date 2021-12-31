package database

import (
	obj "github.com/ubiquitousbyte/wiki-documents/models"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
)

type mongoCategorytStore struct {
	c *mongo.Client
}

var _ CategoryStore = (*mongoCategorytStore)(nil)

func (m *mongoCategorytStore) ReadCategories() ([]obj.Category, error) {
	coll := dbColl(m.c)
	ctx, cancel := dbCtx(1)
	defer cancel()

	filter := bson.M{"name": bson.M{"$exists": true}}
	cursor, err := coll.Find(ctx, filter)
	if err != nil {
		return nil, ErrModelNotFound.from(err)
	}

	var categories []obj.Category
	if err = cursor.All(ctx, &categories); err != nil {
		return nil, ErrInvalidModel.from(err)
	}

	return categories, nil
}

func (m *mongoCategorytStore) ReadCategory(id string) (c obj.Category, err error) {
	coll := dbColl(m.c)
	ctx, cancel := dbCtx(1)
	defer cancel()

	filter := bson.M{"_id": id}
	res := coll.FindOne(ctx, filter)
	if err := res.Err(); err != nil {
		return c, ErrModelNotFound.from(err)
	}

	if err = res.Decode(&c); err != nil {
		return c, ErrInvalidModel.from(err)
	}

	return
}
