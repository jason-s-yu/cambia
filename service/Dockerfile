# service/Dockerfile - multi-stage build for the Go game server
FROM golang:1.26-alpine AS builder

RUN apk add --no-cache git

WORKDIR /build

# Copy go.work and the workspace modules so resolution works. runnerd hosts the
# procmgr package the training dashboard imports; the service build needs it.
COPY go.work go.work
COPY engine/ engine/
COPY runnerd/ runnerd/
COPY service/ service/

# Download deps and build
RUN cd service && go mod download
RUN cd service && CGO_ENABLED=0 GOOS=linux go build -o /cambia-server ./cmd/server/
RUN cd service && CGO_ENABLED=0 GOOS=linux go build -o /cambia-historian ./cmd/db/

# --- Runtime ---
FROM alpine:3.21
RUN apk add --no-cache ca-certificates
COPY --from=builder /cambia-server /usr/local/bin/cambia-server
COPY --from=builder /cambia-historian /usr/local/bin/cambia-historian
# The server also embeds these (see service/migrations/embed.go). The copy stays
# for compose stacks that mount them as docker-entrypoint-initdb.d scripts.
COPY service/migrations/ /app/migrations/

WORKDIR /app
ENV DATA_DIR=/app/data
EXPOSE 8080
CMD ["cambia-server"]
