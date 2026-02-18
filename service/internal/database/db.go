package database

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

var DB *pgxpool.Pool

func ConnectDB() {
	connStr := fmt.Sprintf(
		"postgres://%s:%s@%s:%s/%s",
		os.Getenv("POSTGRES_USER"),
		os.Getenv("POSTGRES_PASSWORD"),
		os.Getenv("PG_HOST"),
		os.Getenv("PG_PORT"),
		os.Getenv("PG_DATABASE"),
	)

	config, err := pgxpool.ParseConfig(connStr)
	if err != nil {
		log.Fatalf("unable to parse pgx config: %v", err)
	}

	DB, err = pgxpool.NewWithConfig(context.Background(), config)
	if err != nil {
		log.Fatalf("unable to create pgx pool: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := DB.Ping(ctx); err != nil {
		log.Fatalf("db ping error: %v", err)
	}

	log.Printf("Connected to database at %s", connStr)
}

// ConnectDBAsync continuously attempts to establish and maintain a database connection.
func ConnectDBAsync() {
	for {
		log.Println("Attempting to connect to database...")
		var err error
		for {
			ConnectDB()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			err = DB.Ping(ctx)
			cancel()
			if err == nil {
				break
			}
			log.Printf("Unable to connect to DB: %v. Retrying in 10 seconds.", err)
			time.Sleep(time.Second * 10)
		}

		// Once connected, periodically check the connection.
		for {
			time.Sleep(time.Minute)
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			err := DB.Ping(ctx)
			cancel()
			if err != nil {
				log.Printf("Lost DB connection: %v. Reconnecting...", err)
				break // exit inner loop to reconnect
			}
		}
	}
}
