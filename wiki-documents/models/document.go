package models

import "go.mongodb.org/mongo-driver/bson/primitive"

type Document struct {
	Id         primitive.ObjectID   `bson:"_id" json:"id"`
	Title      string               `bson:"title" json:"title"`
	Source     string               `bson:"source" json:"source"`
	Categories []primitive.ObjectID `bson:"categories" json:"categories"`
	Paragraphs []Paragraph          `bson:"paragraphs" json:"paragraphs"`
}
