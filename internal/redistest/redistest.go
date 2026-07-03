// Package redistest provides a Redis connection for integration tests,
// mirroring the Python library's testcontainers setup: it uses REDIS_URL
// when set, and otherwise starts a shared Redis 8 container via
// testcontainers-go (skipping tests when Docker is unavailable).
package redistest

import (
	"context"
	"os"
	"sync"
	"testing"

	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/wait"
)

// defaultImage is the pinned Redis image used for integration tests;
// override with the REDIS_IMAGE environment variable.
const defaultImage = "redis:8.8.0"

var (
	once sync.Once
	url  string
	err  error
)

// URL returns a Redis connection URL for integration tests. Priority:
// the REDIS_URL environment variable, then a shared Redis testcontainer
// running defaultImage (started once per test binary; cleaned up by the
// testcontainers reaper). The test is skipped when neither is available.
func URL(t *testing.T) string {
	t.Helper()
	if u := os.Getenv("REDIS_URL"); u != "" {
		return u
	}
	once.Do(func() {
		image := os.Getenv("REDIS_IMAGE")
		if image == "" {
			image = defaultImage
		}
		ctx := context.Background()
		c, e := testcontainers.Run(ctx, image,
			testcontainers.WithExposedPorts("6379/tcp"),
			testcontainers.WithWaitStrategy(wait.ForListeningPort("6379/tcp")),
		)
		if e != nil {
			err = e
			return
		}
		url, err = c.PortEndpoint(ctx, "6379/tcp", "redis")
	})
	if err != nil {
		t.Skipf("redis testcontainer unavailable (set REDIS_URL to use an external server): %v", err)
	}
	return url
}
