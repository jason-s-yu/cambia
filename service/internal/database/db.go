package database

import (
	"context"
	"fmt"
	"log"
	"os"
	"sync"
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

	MigrateIfEnabled()
}

// abandonStaleOnce keeps the boot sweep to a single pass per process, on the same reasoning as
// migrateOnce: ConnectDBAsync re-enters its connect loop whenever the connection drops, and a
// reconnect is not a boot. Running the sweep on a reconnect would abandon the games this very
// process is in the middle of serving.
var abandonStaleOnce sync.Once

// sweepStaleGames closes out games left in progress by a previous process (AbandonStaleGames).
// A failure is logged and not retried: it leaves stale rows in place, which is the state the
// server has always started in, and is no reason to refuse to serve.
func sweepStaleGames() {
	ctx, cancel := context.WithTimeout(context.Background(), staleSweepTimeout)
	defer cancel()

	closed, err := AbandonStaleGames(ctx, DB)
	if err != nil {
		log.Printf("boot sweep: could not abandon games left in progress by a previous process: %v", err)
		return
	}
	if closed > 0 {
		log.Printf("boot sweep: marked %d game(s) left in progress by a previous process as abandoned.", closed)
	}
}

// staleSweepTimeout bounds the boot sweep. It is a single unindexed UPDATE over the games table,
// so it is fast, and a database slow enough to miss this is one the server should get on with
// serving around rather than wait for.
const staleSweepTimeout = 30 * time.Second

// ConnectDBAsync continuously attempts to establish and maintain a database connection.
//
// It also runs the boot sweep that closes out games a previous process left in progress, once,
// after the first connection succeeds. That sits here rather than in ConnectDB because ConnectDB
// is what the historian binary calls too, and the historian must not write games rows
// (cambia-1881).
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

		abandonStaleOnce.Do(sweepStaleGames)

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
