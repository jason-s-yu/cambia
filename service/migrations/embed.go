// Package migrations embeds the SQL schema files so a container image carries
// them in the binary. The same files are still mounted into Postgres as
// docker-entrypoint-initdb.d scripts by the dev and staging compose stacks; the
// embedded copy is what the in-app forward migrator (internal/database) applies
// to an already-initialized database.
package migrations

import "embed"

// FS holds every migration file in this directory, applied in version order by
// database.MigrateIfEnabled.
//
//go:embed *.sql
var FS embed.FS
