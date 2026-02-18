// cmd/server/main.go
package main

import (
	"log"
	"net/http"
	"os"

	"github.com/jason-s-yu/cambia/service/internal/auth"
	"github.com/jason-s-yu/cambia/service/internal/cache"
	"github.com/jason-s-yu/cambia/service/internal/database"
	"github.com/jason-s-yu/cambia/service/internal/handlers"
	"github.com/jason-s-yu/cambia/service/internal/middleware"
	_ "github.com/joho/godotenv/autoload"
	"github.com/sirupsen/logrus"
)

func main() {
	auth.Init()
	go database.ConnectDBAsync()

	// ADDED: Connect to Redis
	if err := cache.ConnectRedis(); err != nil {
		log.Fatalf("Redis connection failed: %v", err)
	}
	log.Println("Redis connected successfully.")

	logger := logrus.New()
	logger.SetLevel(logrus.DebugLevel)

	mux := http.NewServeMux()

	// user endpoints
	mux.HandleFunc("/user/create", handlers.CreateUserHandler)
	mux.HandleFunc("/user/login", handlers.LoginHandler)
	mux.HandleFunc("/user/me", handlers.MeHandler)

	// friend endpoints
	mux.HandleFunc("/friends/add", handlers.AddFriendHandler)
	mux.HandleFunc("/friends/accept", handlers.AcceptFriendHandler)
	mux.HandleFunc("/friends/list", handlers.ListFriendsHandler)
	mux.HandleFunc("/friends/remove", handlers.RemoveFriendHandler)

	// game websocket
	srv := handlers.NewGameServer()

	mux.Handle("/game/ws/", middleware.LogMiddleware(logger)(http.HandlerFunc(
		handlers.GameWSHandler(logger, srv),
	)))

	// lobby endpoints
	mux.Handle("/lobby/create", middleware.LogMiddleware(logger)(http.HandlerFunc(
		handlers.CreateLobbyHandler(srv),
	)))
	mux.Handle("/lobby/list", middleware.LogMiddleware(logger)(http.HandlerFunc(
		handlers.ListLobbiesHandler(srv),
	)))

	// lobby ws
	mux.Handle("/lobby/ws/", middleware.LogMiddleware(logger)(http.HandlerFunc(
		handlers.LobbyWSHandler(logger, srv),
	)))

	addr := ":8080"
	if port := os.Getenv("PORT"); port != "" {
		addr = ":" + port
	}
	logger.Infof("Running on %s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatalf("server exited: %v", err)
	}
}
