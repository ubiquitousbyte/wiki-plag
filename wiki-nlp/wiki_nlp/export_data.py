import pymongo

if __name__ == '__main__':
    uri = f"mongodb://wikiplag:wikiplag2021@localhost:27017/wikiplag"
    client = pymongo.MongoClient(uri)
    pipeline = [
        # Stage 1
        # Flatten all paragraph fields
        {
            "$unwind": {"path": "$paragraphs"}
        },

        # Stage 2
        # Remove all paragraphs that do not have any text in them
        {
            "$match": {
                "paragraphs.text": {"$ne": ""},
                "paragraphs.title": {
                    "$nin": ['Weblinks', 'Literatur', 'Siehe auch']
                }
            }
        },
        # Stage 3
        # Project the required fields into a new collection
        {
            "$project": {
                "_id": 0,
                "document": {
                    "id": "$_id",
                    "title": "$title"
                },
                "title": "$paragraphs.title",
                "text": "$paragraphs.text",
                "position": "$paragraphs.position"
            }
        },
        # Stage 4
        # Load results into the collection
        {
            "$out": "nlp"
        }
    ]
    input_coll = client["wikiplag"]["documents"]
    input_coll.aggregate(pipeline)

    output_coll = client["wikiplag"]["nlp"]
    output_coll.create_index("index")

    for i, entry in enumerate(output_coll.find()):
        output_coll.update_one({"_id": entry['_id']}, {"$set": {"index": i}})

        # Assign each entry a unique mapping to the document matrix of the model
    # for i, entry in enumerate(coll.find()):
    #    coll.update_one({"_id": entry['_id']}, {"$set": {"dmmIndex": i}})
