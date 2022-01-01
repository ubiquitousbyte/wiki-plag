package entity

type Paragraph struct {
	Title    string `bson:"title" json:"title"`
	Position int    `bson:"position" json:"position"`
	Text     string `bson:"text" json:"text"`
}

type Document struct {
	Id         Id          `bson:"_id" json:"id"`
	Title      string      `bson:"title" json:"title"`
	Source     string      `bson:"source" json:"source"`
	Categories []Id        `bson:"categories" json:"categories"`
	Paragraphs []Paragraph `bson:"paragraphs" json:"paragraphs"`
}
