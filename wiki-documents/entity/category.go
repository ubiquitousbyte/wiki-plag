package entity

type Category struct {
	Id          Id     `bson:"_id,omitempty" json:"id"`
	Source      string `bson:"source" json:"source"`
	Name        string `bson:"name" json:"name"`
	Description string `bson:"description" json:"description"`
}
