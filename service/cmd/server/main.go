// cmd/server/main.go
package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/procmgr"
	"github.com/jason-s-yu/cambia/service/internal/auth"
	"github.com/jason-s-yu/cambia/service/internal/cache"
	"github.com/jason-s-yu/cambia/service/internal/database"
	"github.com/jason-s-yu/cambia/service/internal/handlers"
	"github.com/jason-s-yu/cambia/service/internal/harnessproxy"
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

	// Connect to Redis. Non-fatal (CF#3): the dashboard and core server run
	// without it; only the game-action historian queue degrades.
	if err := cache.ConnectRedis(); err != nil {
		log.Printf("Redis connection failed (non-fatal): %v", err)
	} else {
		log.Println("Redis connected successfully.")
	}

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

	// Training dashboard routes. All /training/* and /ws/training/* endpoints
	// are gated by RequireAuth (CF#1), logged by LogMiddleware.
	runsDir := os.Getenv("CAMBIA_RUNS_DIR")
	if runsDir == "" {
		runsDir = "../cfr/runs"
	}
	cfrDir := os.Getenv("CAMBIA_CFR_DIR")
	if cfrDir == "" {
		cfrDir = "../cfr"
	}
	cambiaBin := os.Getenv("CAMBIA_BIN")
	if cambiaBin == "" {
		cambiaBin = "cambia"
	}
	maxConcurrent := 1
	if v := os.Getenv("CAMBIA_MAX_CONCURRENT_RUNS"); v != "" {
		if n, perr := strconv.Atoi(v); perr == nil && n > 0 {
			maxConcurrent = n
		}
	}
	// minVRAMGB/minDiskGB are 0 unless the env var overrides them, which tells
	// NewProcessHandlers to fall back to the contract defaults (4 GiB / 5 GiB) --
	// the preflight rails must not be left off by an empty environment.
	var minVRAMGB, minDiskGB float64
	if v := os.Getenv("CAMBIA_MIN_FREE_VRAM_GB"); v != "" {
		if f, perr := strconv.ParseFloat(v, 64); perr == nil && f > 0 {
			minVRAMGB = f
		}
	}
	if v := os.Getenv("CAMBIA_MIN_FREE_DISK_GB"); v != "" {
		if f, perr := strconv.ParseFloat(v, 64); perr == nil && f > 0 {
			minDiskGB = f
		}
	}
	trainingStore, err := training.NewTrainingStore(runsDir)
	if err != nil {
		log.Printf("Training store init failed (non-fatal): %v", err)
	} else {
		defer trainingStore.Close()

		// authWrap composes the request logger over the JWT auth gate.
		authWrap := func(h http.Handler) http.Handler {
			return middleware.LogMiddleware(logger)(middleware.RequireAuth(h))
		}

		procMgr := procmgr.NewProcessManager(runsDir, cfrDir, cambiaBin, trainingStore, procmgr.TrainAlgorithms())
		procMgr.SetMaxConcurrent(maxConcurrent)
		procMgr.Reconcile()
		procHandlers := training.NewProcessHandlers(training.ProcessHandlersConfig{
			Manager:       procMgr,
			Store:         trainingStore,
			CambiaBin:     cambiaBin,
			CFRDir:        cfrDir,
			RunsDir:       runsDir,
			TemplateDir:   filepath.Join(cfrDir, "config"),
			MaxConcurrent: maxConcurrent,
			MinVRAMGB:     minVRAMGB,
			MinDiskGB:     minDiskGB,
		})

		// Harness control-plane proxy (cambia-295 v1.1): loaded from the
		// host-local harness config. Absent config -> nil client -> remote runs
		// stay read-only (409) and log tails use the synced file (exact v1
		// behavior). A present-but-invalid config or a client-init failure is
		// logged and treated as absent, never fatal.
		if hcfg, herr := harnessproxy.LoadConfig(); herr == nil {
			if client, cerr := harnessproxy.New(hcfg); cerr == nil {
				trainingStore.SetHarnessProxy(client)
				procHandlers.SetHarnessProxy(client)
				log.Printf("Harness proxy enabled: runner=%s origin=%s", hcfg.RunnerURL, hcfg.OriginHost)
			} else {
				log.Printf("Harness proxy config found but client init failed (remote runs read-only): %v", cerr)
			}
		} else if !errors.Is(herr, harnessproxy.ErrNoConfig) {
			log.Printf("Harness proxy config error (remote runs read-only): %v", herr)
		}

		// Eval subsystem: a supervised `cambia evaluate` child with its own
		// in-memory job registry and concurrency cap (default 1), separate from the
		// training cap in ProcessManager. Same preflight defaults as ProcessHandlers.
		maxConcurrentEvals := 1
		if v := os.Getenv("CAMBIA_MAX_CONCURRENT_EVALS"); v != "" {
			if n, perr := strconv.Atoi(v); perr == nil && n > 0 {
				maxConcurrentEvals = n
			}
		}
		evalMgr := training.NewEvalManager(runsDir, cfrDir, cambiaBin)
		evalMgr.SetMaxConcurrent(maxConcurrentEvals)
		evalHandlers := training.NewEvalHandlers(training.EvalHandlersConfig{
			Manager:   evalMgr,
			RunsDir:   runsDir,
			MinVRAMGB: minVRAMGB,
			MinDiskGB: minDiskGB,
		})

		// Host resource sampler: one server-side poller with WS fan-out. A
		// non-positive CAMBIA_RESOURCE_POLL_SEC falls back to the 2s default.
		resourcePoll := 2 * time.Second
		if v := os.Getenv("CAMBIA_RESOURCE_POLL_SEC"); v != "" {
			if n, perr := strconv.Atoi(v); perr == nil && n > 0 {
				resourcePoll = time.Duration(n) * time.Second
			}
		}
		resMon := training.NewResourceMonitor(runsDir, resourcePoll)
		defer resMon.Close()

		registerTrainingRoutes(mux, authWrap, trainingStore, procHandlers, evalHandlers, resMon)
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

// registerTrainingRoutes wires every /training and /ws/training endpoint onto mux
// behind authWrap. It is split out of main so the routing rules -- the
// /training/runs sub-router (including the eval GET/POST case), the resources and
// compare endpoints, and the /ws/training resources-vs-logs disambiguation -- are
// exercised by a wiring test without standing up the full server.
func registerTrainingRoutes(
	mux *http.ServeMux,
	authWrap func(http.Handler) http.Handler,
	trainingStore *training.TrainingStore,
	procHandlers *training.ProcessHandlers,
	evalHandlers *training.EvalHandlers,
	resMon *training.ResourceMonitor,
) {
	// GET list / POST create.
	mux.Handle("/training/runs", authWrap(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			trainingStore.HandleListRuns(w, r)
		case http.MethodPost:
			procHandlers.HandleCreate(w, r)
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	})))

	// GET detail/metrics/checkpoints/eval; POST start/stop/resume/eval.
	mux.Handle("/training/runs/", authWrap(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
		if len(parts) == 3 {
			trainingStore.HandleGetRun(w, r)
			return
		}
		if len(parts) == 4 {
			switch parts[3] {
			case "metrics":
				trainingStore.HandleGetMetrics(w, r)
			case "checkpoints":
				trainingStore.HandleGetCheckpoints(w, r)
			case "eval":
				switch r.Method {
				case http.MethodGet:
					evalHandlers.HandleList(w, r)
				case http.MethodPost:
					evalHandlers.HandleTrigger(w, r)
				default:
					http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
				}
			case "start":
				procHandlers.HandleStart(w, r)
			case "stop":
				procHandlers.HandleStop(w, r)
			case "resume":
				procHandlers.HandleResume(w, r)
			default:
				http.NotFound(w, r)
			}
			return
		}
		http.NotFound(w, r)
	})))

	// GET config template listing.
	mux.Handle("/training/config/templates", authWrap(http.HandlerFunc(procHandlers.HandleTemplates)))

	// GET one-off host resource snapshot.
	mux.Handle("/training/system/resources", authWrap(http.HandlerFunc(resMon.HandleSnapshot)))

	// GET run comparison (?runs=a,b,c).
	mux.Handle("/training/compare", authWrap(http.HandlerFunc(trainingStore.HandleCompare)))

	// WS disambiguation: /ws/training/resources is the resource stream; every other
	// /ws/training/{name}/logs path is the per-run log streamer. The exact-match must
	// come first or the log streamer swallows the resources path (extractRunName
	// would otherwise treat "resources" as a run name).
	mux.Handle("/ws/training/", authWrap(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/ws/training/resources" {
			resMon.HandleWS(w, r)
			return
		}
		trainingStore.HandleLogStream(w, r)
	})))
}
