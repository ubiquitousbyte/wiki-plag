// Package models contains application-specific entities
package models

type Paragraph struct {
	Title    string `bson:"title" json:"title"`
	Position int    `bson:"position" json:"position"`
	Text     string `bson:"text" json:"text"`
}
