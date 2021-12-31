package database

import (
	"context"
	"time"

	obj "github.com/ubiquitousbyte/wiki-documents/models"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
)

func dbColl(cl *mongo.Client) *mongo.Collection {
	return cl.Database("wikiplag").Collection("documents")
}

func dbCtx(seconds time.Duration) (context.Context, context.CancelFunc) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*seconds)
	return ctx, cancel
}

type mongoDocStore struct {
	client *mongo.Client
}

var _ DocumentStore = (*mongoDocStore)(nil)

func (m *mongoDocStore) ReadDocsByCategory(categoryId string) ([]obj.Document, error) {
	catObjId, err := primitive.ObjectIDFromHex(categoryId)
	if err != nil {
		return nil, ErrInvalidModel.from(err)
	}

	coll := dbColl(m.client)
	ctx, cancel := dbCtx(2)
	defer cancel()

	filter := bson.M{"categories": bson.M{"$in": catObjId}}
	cursor, err := coll.Find(ctx, filter)
	if err != nil {
		return nil, ErrModelNotFound.from(err)
	}

	var documents []obj.Document
	if err = cursor.All(ctx, &documents); err != nil {
		return nil, ErrInvalidModel.from(err)
	}

	return documents, nil
}

func (m *mongoDocStore) ReadDoc(id string) (doc obj.Document, err error) {
	coll := dbColl(m.client)
	ctx, cancel := dbCtx(1)
	defer cancel()

	filter := bson.M{"_id": id}
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

func (m *mongoDocStore) ReadDocBySrc(title, source string) (obj.Document, error) {
	var doc obj.Document

	coll := dbColl(m.client)
	ctx, cancel := dbCtx(1)
	defer cancel()

	filter := bson.M{"title": title, source: "source"}
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

func (m *mongoDocStore) CreateDoc(doc *obj.Document) (id string, err error) {
	coll := dbColl(m.client)
	ctx, cancel := dbCtx(1)
	defer cancel()

	res, err := coll.InsertOne(ctx, doc)
	if err != nil {
		return id, ErrCreate.from(err)
	}

	id = res.InsertedID.(primitive.ObjectID).Hex()
	return
}

func (m *mongoDocStore) AddCategory(docId, categoryId string) error {
	docObjId, err := primitive.ObjectIDFromHex(docId)
	if err != nil {
		return ErrInvalidModel.from(err)
	}

	catObjId, err := primitive.ObjectIDFromHex(categoryId)
	if err != nil {
		return ErrInvalidModel.from(err)
	}

	coll := dbColl(m.client)
	ctx, cancel := dbCtx(1)
	defer cancel()

	filter := bson.M{"_id": docObjId}
	update := bson.M{"$addToSet": bson.M{"categories": catObjId}}
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

func (m *mongoDocStore) RemoveCategory(docId, categoryId string) error {
	docObjId, err := primitive.ObjectIDFromHex(docId)
	if err != nil {
		return ErrInvalidModel.from(err)
	}

	catObjId, err := primitive.ObjectIDFromHex(categoryId)
	if err != nil {
		return ErrInvalidModel.from(err)
	}

	coll := dbColl(m.client)
	ctx, cancel := dbCtx(1)
	defer cancel()

	filter := bson.M{"_id": docObjId}
	update := bson.M{
		"$pull": bson.M{
			"categories": bson.M{"$in": []primitive.ObjectID{catObjId}},
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

func (m *mongoDocStore) DeleteDoc(id string) error {
	coll := dbColl(m.client)
	ctx, cancel := dbCtx(1)
	defer cancel()

	filter := bson.M{"_id": id}
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
