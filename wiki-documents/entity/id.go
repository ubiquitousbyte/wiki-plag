package entity

import (
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/bsontype"
	"go.mongodb.org/mongo-driver/bson/primitive"
)

type Id string

func (id Id) MarshalBSONValue() (bsontype.Type, []byte, error) {
	p, err := primitive.ObjectIDFromHex(string(id))
	if err != nil {
		return bsontype.Null, nil, err
	}
	return bson.MarshalValue(p)
}

func (id Id) IsValidId() bool {
	return primitive.IsValidObjectID(string(id))
}

func NewEntityId() Id {
	return Id(primitive.NewObjectID().Hex())
}

func (id Id) String() string {
	return string(id)
}
