// internal/middleware/logging.go

package middleware

import (
	"net/http"
	"time"

	"github.com/sirupsen/logrus"
)

// LogMiddleware is an HTTP middleware that logs incoming requests using Logrus.
// Logs the method, path, and duration of each request.
func LogMiddleware(logger *logrus.Logger) func(next http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			path := r.URL.Path
			method := r.Method

			next.ServeHTTP(w, r)

			duration := time.Since(start)
			logger.WithFields(logrus.Fields{
				"method":   method,
				"path":     path,
				"duration": duration,
				"remote":   r.RemoteAddr,
			}).Info()
		})
	}
}

// LogWebSocketConnect logs a message when a WebSocket client connects.
// Typically called in your WebSocket handler once you accept an upgrade.
func LogWebSocketConnect(logger *logrus.Logger, remoteAddr string, path string) {
	logger.WithFields(logrus.Fields{
		"remote": remoteAddr,
		"path":   path,
	}).Info("WebSocket connected")
}

// LogWebSocketDisconnect logs a message when a WebSocket client disconnects.
func LogWebSocketDisconnect(logger *logrus.Logger, remoteAddr string, path string, err error) {
	fields := logrus.Fields{
		"remote": remoteAddr,
		"path":   path,
	}
	if err != nil {
		fields["error"] = err
	}
	logger.WithFields(fields).Info("WebSocket disconnected")
}
