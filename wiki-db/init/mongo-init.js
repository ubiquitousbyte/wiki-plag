db.log.insertOne({ "message": "Database created." });
try {
    db.createUser(
        {
            user: _getEnv("MONGO_USER"),
            pwd: cat(_getEnv("MONGO_PASSWORD_FILE")),
            roles: [
                "readWrite", "dbAdmin"
            ]
        }
    );
} catch (error) {
    db.log.insertOne({ "message": error })
}
