// cmd/server/main.go
package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/jason-s-yu/cambia/service/internal/auth"
	"github.com/jason-s-yu/cambia/service/internal/cache"
	"github.com/jason-s-yu/cambia/service/internal/database"
	"github.com/jason-s-yu/cambia/service/internal/handlers"
	"github.com/jason-s-yu/cambia/service/internal/hub"
	"github.com/jason-s-yu/cambia/service/internal/matchmaking"
	"github.com/jason-s-yu/cambia/service/internal/middleware"
	"github.com/jason-s-yu/cambia/service/internal/training"
	_ "github.com/joho/godotenv/autoload"
	"github.com/sirupsen/logrus"
)

func main() {
	dataDir := os.Getenv("DATA_DIR")
	if dataDir == "" {
		dataDir = "./data"
	}
	privPath := filepath.Join(dataDir, "jwt_keys", "private.pem")
	pubPath := filepath.Join(dataDir, "jwt_keys", "public.pem")
	if err := auth.InitFromPath(privPath, pubPath); err != nil {
		if err2 := auth.InitAndSave(dataDir); err2 != nil {
			log.Fatalf("failed to init JWT keys: %v", err2)
		}
		log.Println("Generated new JWT key pair and saved to disk.")
	} else {
		log.Println("Loaded JWT keys from disk.")
	}
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
	mux.HandleFunc("/user/guest", handlers.GuestHandler)
	mux.HandleFunc("/user/login", handlers.LoginHandler)
	mux.HandleFunc("/user/logout", handlers.LogoutHandler)
	mux.HandleFunc("/user/me", handlers.MeHandler)
	mux.HandleFunc("/user/claim", handlers.ClaimEphemeralHandler)

	// friend endpoints
	mux.HandleFunc("/friends/add", handlers.AddFriendHandler)
	mux.HandleFunc("/friends/accept", handlers.AcceptFriendHandler)
	mux.HandleFunc("/friends/list", handlers.ListFriendsHandler)
	mux.HandleFunc("/friends/remove", handlers.RemoveFriendHandler)

	srv := handlers.NewGameServer()

	// Wire matchmaker callback before starting Run.
	srv.Matchmaker.OnMatchFormed = func(result matchmaking.MatchResult) {
		h, ok := srv.HubStore.GetHub(result.HostLobbyID)
		if !ok {
			log.Printf("Match formed but host hub %s not found", result.HostLobbyID)
			return
		}
		players := make([]hub.MatchedPlayer, len(result.Players))
		for i, p := range result.Players {
			players[i] = hub.MatchedPlayer{
				UserID:   p.UserID,
				Username: p.Username,
			}
		}
		h.Matched() <- players
	}

	go srv.Matchmaker.Run(context.Background())

	// unified WebSocket endpoint
	mux.Handle("/ws/", middleware.LogMiddleware(logger)(http.HandlerFunc(
		handlers.HubWSHandler(logger, srv),
	)))

	// lobby endpoints
	mux.Handle("/lobby/create", middleware.LogMiddleware(logger)(http.HandlerFunc(
		handlers.CreateLobbyHandler(srv),
	)))
	mux.Handle("/lobby/list", middleware.LogMiddleware(logger)(http.HandlerFunc(
		handlers.ListLobbiesHandler(srv),
	)))

	// lobby action router (join, search)
	mux.Handle("/lobby/", middleware.LogMiddleware(logger)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
		if len(parts) >= 3 {
			switch parts[2] {
			case "search":
				if r.Method == http.MethodPost {
					handlers.SearchLobbyHandler(srv)(w, r)
				} else if r.Method == http.MethodDelete {
					handlers.CancelSearchHandler(srv)(w, r)
				} else {
					http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
				}
				return
			case "join":
				handlers.JoinLobbyHandler(srv)(w, r)
				return
			}
		}
		http.NotFound(w, r)
	})))

	// matchmaking queue list
	mux.Handle("/matchmaking/queues", middleware.LogMiddleware(logger)(http.HandlerFunc(
		handlers.ListQueuesHandler(srv),
	)))

	// Training dashboard routes
	runsDir := os.Getenv("CAMBIA_RUNS_DIR")
	if runsDir == "" {
		runsDir = "../cfr/runs"
	}
	trainingStore, err := training.NewTrainingStore(runsDir)
	if err != nil {
		log.Printf("Training store init failed (non-fatal): %v", err)
	} else {
		defer trainingStore.Close()
		mux.Handle("/training/runs", middleware.LogMiddleware(logger)(http.HandlerFunc(trainingStore.HandleListRuns)))
		mux.Handle("/training/runs/", middleware.LogMiddleware(logger)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
			if len(parts) == 3 {
				trainingStore.HandleGetRun(w, r)
			} else if len(parts) == 4 {
				switch parts[3] {
				case "metrics":
					trainingStore.HandleGetMetrics(w, r)
				case "checkpoints":
					trainingStore.HandleGetCheckpoints(w, r)
				default:
					http.NotFound(w, r)
				}
			} else {
				http.NotFound(w, r)
			}
		})))
		mux.Handle("/ws/training/", middleware.LogMiddleware(logger)(http.HandlerFunc(trainingStore.HandleLogStream)))
	}

	addr := ":8080"
	if port := os.Getenv("PORT"); port != "" {
		addr = ":" + port
	}
	logger.Infof("Running on %s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatalf("server exited: %v", err)
	}
}
