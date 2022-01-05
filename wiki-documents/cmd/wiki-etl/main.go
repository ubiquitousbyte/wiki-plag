package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path"
	"time"

	"github.com/ubiquitousbyte/wiki-documents/crawler"
	"github.com/ubiquitousbyte/wiki-documents/database"
	"github.com/ubiquitousbyte/wiki-documents/entity"
	"github.com/ubiquitousbyte/wiki-documents/etl"
	"github.com/ubiquitousbyte/wiki-documents/mediawiki"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

type Duration struct {
	time.Duration
}

func (d *Duration) UnmarshalJSON(b []byte) (err error) {
	if b[0] == '"' {
		sd := string(b[1 : len(b)-1])
		d.Duration, err = time.ParseDuration(sd)
		return
	}
	var id int64
	id, err = json.Number(string(b)).Int64()
	d.Duration = time.Duration(id)
	return
}

type DatabaseConfig struct {
	Uri      string `json:"uri"`      // The database URI to connect to
	User     string `json:"user"`     // The database user
	Password string `json:"password"` // The user password
}

type EtlConfig struct {
	Language string          `json:"language"` // The language subsystem to extract from
	Category entity.Category `json:"root"`     // The root category to start extracting from
	Duration Duration        `json:"duration"` // How long to extract for
	Interval Duration        `json:"interval"` // Time slice between node traversal
}

type Config struct {
	Etl EtlConfig      `json:"etl"`
	Db  DatabaseConfig `json:"database"`
}

func main() {
	currentDir, err := os.Getwd()
	if err != nil {
		fmt.Println(err)
		return
	}

	defaultPath := path.Join(currentDir, "etl-config.json")
	path := flag.String("cfg", defaultPath,
		"Path to etl configuration file."+
			"If omitted, the application tries to parse an etl-config.json file"+
			"in the current directory.")
	flag.Parse()

	contents, err := os.ReadFile(*path)
	if err != nil {
		fmt.Println(err)
		return
	}

	var config Config
	if err = json.Unmarshal(contents, &config); err != nil {
		fmt.Println(err)
		return
	}

	auth := options.Credential{
		AuthSource: "wikiplag",
		Username:   config.Db.User,
		Password:   config.Db.Password,
	}

	dbOpts := options.Client().ApplyURI(config.Db.Uri).SetAuth(auth)
	client, err := mongo.Connect(context.Background(), dbOpts)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer client.Disconnect(context.Background())

	if err = client.Ping(context.Background(), nil); err != nil {
		fmt.Println(err)
		return
	}

	crawlCfg := crawler.Config{
		Language: config.Etl.Language,
		MwC:      mediawiki.NewClient(),
		Logger:   log.Default(),
	}

	documentStore := database.NewMongoDocumentStore(client)
	categoryStore := database.NewMongoCategoryStore(client)

	pipeline := etl.NewPipeline(etl.Config{
		CrawlCfg: crawlCfg,
		Ds:       documentStore,
		Cs:       categoryStore,
	})

	err = pipeline.BFS(config.Etl.Duration.Duration, config.Etl.Interval.Duration, &config.Etl.Category)
	if err != nil {
		fmt.Println(err)
		return
	}
}
