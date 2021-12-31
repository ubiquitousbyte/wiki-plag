package models

import "go.mongodb.org/mongo-driver/bson/primitive"

type Category struct {
	Id          primitive.ObjectID `bson:"id" json:"id"`
	Source      string             `bson:"source" json:"source"`
	Name        string             `bson:"name" json:"name"`
	Description string             `bson:"description" json:"description"`
}
